import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
import random
import json
import math
from functools import partial
from typing import Iterator, List, Optional, Sequence, Tuple
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
import gc
from optimizers import StableAdamW
from utils import setup_seed, get_logger, ader_evaluator

# Dataset-Related Modules
from mvtec2_dataset import MVTec2Dataset
from mvtec2_dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from utils import get_gaussian_kernel, cal_anomaly_maps

# Model-Related Modules
from models import vit_encoder
from models.uad2 import INP_Former, ResidualSegHead
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from PIL import Image, ImageEnhance
import cv2


warnings.filterwarnings("ignore")

CROP_COORD_PATTERN = re.compile(r"^(?P<base>.+)_x(?P<x1>\d+)_y(?P<y1>\d+)_x(?P<x2>\d+)_y(?P<y2>\d+)$")
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

def update_final_thresholds_by_split_json(
    category_name,
    normal_row,
    save_path,
    test_img_dirs,
    mixed_row=None,
    mixed_split_name="test_private_mixed",
    threshold_key="final_threshold_255",
):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if os.path.isfile(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            try:
                result = json.load(f)
            except json.JSONDecodeError:
                result = {}
    else:
        result = {}

    base_threshold = float(normal_row[threshold_key])

    if mixed_row is None or mixed_row.get(threshold_key) is None:
        mixed_threshold = base_threshold
    else:
        mixed_threshold = float(mixed_row[threshold_key])

    result[str(category_name)] = {}
    for split_name in test_img_dirs:
        if split_name == mixed_split_name:
            result[str(category_name)][split_name] = mixed_threshold
        else:
            result[str(category_name)][split_name] = base_threshold

    tmp_save_path = save_path + ".tmp"
    with open(tmp_save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.replace(tmp_save_path, save_path)

def compose_anomaly_map_with_seg(recon_map, seg_logits, seg_weight):
    if seg_logits is None or seg_weight <= 0:
        return recon_map

    seg_prob = torch.sigmoid(seg_logits)

    if seg_prob.shape[-2:] != recon_map.shape[-2:]:
        seg_prob = F.interpolate(seg_prob, size=recon_map.shape[-2:], mode='bilinear', align_corners=False)

    return torch.clamp((1.0 - seg_weight) * recon_map + seg_weight * seg_prob, 0.0, 1.0)


def is_dinov3_encoder(encoder_name: str) -> bool:
    return encoder_name.startswith('dinov3_')


def make_divisible_down(x: int, divisor: int) -> int:
    return max(divisor, (x // divisor) * divisor)


def get_encoder_config(encoder_name: str):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]

    if encoder_name in {'dinov3_vits16', 'dinov3_vits16plus'}:
        return 384, 6, target_layers
    if encoder_name == 'dinov3_vitb16':
        return 768, 12, target_layers
    if encoder_name in {'dinov3_vitl16', 'dinov3_vitl16plus'}:
        return 1024, 16, [4, 6, 8, 10, 12, 14, 16, 18]
    if encoder_name == 'dinov3_vith16plus':
        return 1280, 20, [8, 11, 14, 17, 20, 23, 26, 29]

    if 'small' in encoder_name:
        return 384, 6, target_layers
    if 'base' in encoder_name:
        return 768, 12, target_layers
    if 'large' in encoder_name:
        return 1024, 16, [4, 6, 8, 10, 12, 14, 16, 18]

    raise RuntimeError('Architecture not in supported small/base/large or DINOv3 families.')


def parse_crop_coordinate_from_path(img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    match = CROP_COORD_PATTERN.match(stem)

    if match is None:
        return stem, None

    base_name = match.group("base")
    x1 = int(match.group("x1"))
    y1 = int(match.group("y1"))
    x2 = int(match.group("x2"))
    y2 = int(match.group("y2"))

    return base_name, (x1, y1, x2, y2)


def get_merge_key_from_path(img_path):
    base_name, coord = parse_crop_coordinate_from_path(img_path)
    img_dir = os.path.dirname(img_path)
    return os.path.join(img_dir, base_name)


def get_merge_sort_key_from_path(img_path):
    base_name, coord = parse_crop_coordinate_from_path(img_path)
    img_dir = os.path.dirname(img_path)

    if coord is None:
        return img_dir, base_name, 0, 0, 0, 0, img_path

    x1, y1, x2, y2 = coord
    return img_dir, base_name, y1, x1, y2, x2, img_path


def extract_image_paths_from_dataset(dataset):
    candidate_attrs = [
        "img_paths",
        "image_paths",
        "img_path",
        "image_path",
        "img_path_list",
        "image_path_list",
        "paths",
        "path_list",
        "imgs",
        "samples",
        "data",
    ]

    dataset_len = len(dataset)

    for attr_name in candidate_attrs:
        if not hasattr(dataset, attr_name):
            continue

        value = getattr(dataset, attr_name)

        if not isinstance(value, (list, tuple)):
            continue

        if len(value) != dataset_len:
            continue

        if len(value) == 0:
            return []

        first_item = value[0]

        if isinstance(first_item, str):
            return list(value)

        if isinstance(first_item, (list, tuple)) and len(first_item) > 0 and isinstance(first_item[0], str):
            return [x[0] for x in value]

    return None


class SortedByMergeKeyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        image_paths = extract_image_paths_from_dataset(dataset)

        if image_paths is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = sorted(
                range(len(dataset)),
                key=lambda idx: get_merge_sort_key_from_path(image_paths[idx]),
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def normalize_mask_to_uint8(gt_mask):
    gt_mask = np.asarray(gt_mask)
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[:, :, 0]
    if gt_mask.max() <= 1.0:
        gt_mask = gt_mask * 255.0
    return ((gt_mask > 127).astype(np.uint8) * 255)


def four_tuple_collate(batch):
    imgs, gts, labels, paths = zip(*batch)
    imgs = torch.stack(list(imgs), dim=0)
    gts = torch.stack(list(gts), dim=0)
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    paths = list(paths)
    return imgs, gts, labels, paths


def get_crop_original_shape(img_path):
    with Image.open(img_path) as im:
        w, h = im.size
    return h, w


def get_merge_key_and_coord(img_path, patch_shape):
    base_name, coord = parse_crop_coordinate_from_path(img_path)
    img_dir = os.path.dirname(img_path)

    if coord is None:
        h, w = patch_shape[:2]
        coord = (0, 0, w - 1, h - 1)

    merge_key = os.path.join(img_dir, base_name)

    return merge_key, base_name, coord


def get_valid_crop_size_from_path(img_path, fallback_shape):
    _, coord = parse_crop_coordinate_from_path(img_path)

    if coord is not None:
        x1, y1, x2, y2 = coord
        return y2 - y1 + 1, x2 - x1 + 1

    return get_crop_original_shape(img_path)


def resize_merge_record(record, new_h, new_w):
    old_h, old_w = record["accumulator"].shape[:2]

    if new_h <= old_h and new_w <= old_w:
        return record

    target_h = max(old_h, new_h)
    target_w = max(old_w, new_w)

    new_accumulator = np.zeros((target_h, target_w), dtype=np.float32)
    new_counter = np.zeros((target_h, target_w), dtype=np.uint16)
    new_gt = np.zeros((target_h, target_w), dtype=np.uint8)

    new_accumulator[:old_h, :old_w] = record["accumulator"]
    new_counter[:old_h, :old_w] = record["counter"]
    new_gt[:old_h, :old_w] = record["gt"]

    record["accumulator"] = new_accumulator
    record["counter"] = new_counter
    record["gt"] = new_gt

    return record


def get_gt_mask_from_tensor(gt_tensor, idx, target_h, target_w):
    gt_item = gt_tensor[idx].detach().cpu().numpy()

    if gt_item.ndim == 3:
        if gt_item.shape[0] == 1:
            gt_item = gt_item[0]
        else:
            gt_item = np.max(gt_item, axis=0)

    if gt_item.max() <= 1.0:
        gt_item = gt_item * 255.0

    gt_item = cv2.resize(gt_item.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    gt_item = ((gt_item > 127).astype(np.uint8) * 255)

    return gt_item


def add_crop_to_merge_record(record, img_path, anomaly_map, gt_mask, merge_mode="average"):
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    gt_mask = normalize_mask_to_uint8(gt_mask)

    merge_key, base_name, coord = get_merge_key_and_coord(img_path, anomaly_map.shape)
    x1, y1, x2, y2 = coord

    target_h = y2 + 1
    target_w = x2 + 1

    if record is None:
        record = {
            "key": merge_key,
            "base_name": base_name,
            "first_img_path": img_path,
            "accumulator": np.zeros((target_h, target_w), dtype=np.float32),
            "counter": np.zeros((target_h, target_w), dtype=np.uint16),
            "gt": np.zeros((target_h, target_w), dtype=np.uint8),
        }
    else:
        record = resize_merge_record(record, target_h, target_w)

    valid_h = min(y2 - y1 + 1, anomaly_map.shape[0], record["accumulator"].shape[0] - y1)
    valid_w = min(x2 - x1 + 1, anomaly_map.shape[1], record["accumulator"].shape[1] - x1)

    if valid_h <= 0 or valid_w <= 0:
        return record

    anomaly_valid = anomaly_map[:valid_h, :valid_w]
    gt_valid = gt_mask[:valid_h, :valid_w]

    if merge_mode == "average":
        record["accumulator"][y1:y1 + valid_h, x1:x1 + valid_w] += anomaly_valid
        record["counter"][y1:y1 + valid_h, x1:x1 + valid_w] += 1
    elif merge_mode == "max":
        region = record["accumulator"][y1:y1 + valid_h, x1:x1 + valid_w]
        covered = record["counter"][y1:y1 + valid_h, x1:x1 + valid_w] > 0

        region[covered] = np.maximum(region[covered], anomaly_valid[covered])
        region[~covered] = anomaly_valid[~covered]

        record["accumulator"][y1:y1 + valid_h, x1:x1 + valid_w] = region
        record["counter"][y1:y1 + valid_h, x1:x1 + valid_w] = 1
    elif merge_mode == "overwrite":
        record["accumulator"][y1:y1 + valid_h, x1:x1 + valid_w] = anomaly_valid
        record["counter"][y1:y1 + valid_h, x1:x1 + valid_w] = 1
    else:
        raise ValueError(f"Unsupported merge_mode: {merge_mode}")

    record["gt"][y1:y1 + valid_h, x1:x1 + valid_w] = np.maximum(
        record["gt"][y1:y1 + valid_h, x1:x1 + valid_w],
        gt_valid,
    )

    return record


def finalize_single_merge_record(record, merge_mode="average"):
    counter = record["counter"]

    if merge_mode == "average":
        anomaly_map = record["accumulator"] / np.maximum(counter.astype(np.float32), 1.0)
    else:
        anomaly_map = record["accumulator"]

    valid_region = counter > 0

    if np.any(valid_region):
        rows = np.where(np.any(valid_region, axis=1))[0]
        cols = np.where(np.any(valid_region, axis=0))[0]
        y1, y2 = rows[0], rows[-1] + 1
        x1, x2 = cols[0], cols[-1] + 1
        anomaly_map = anomaly_map[y1:y2, x1:x2]
        gt_mask = record["gt"][y1:y2, x1:x2]
    else:
        gt_mask = record["gt"]

    merged_record = {
        "key": record["key"],
        "base_name": record["base_name"],
        "first_img_path": record["first_img_path"],
        "anomaly_map": np.clip(anomaly_map, 0, 1).astype(np.float32),
        "gt_mask": normalize_mask_to_uint8(gt_mask),
    }

    return merged_record


def resize_merged_record_for_eval(anomaly_map, gt_mask, resize_mask=None):
    anomaly_map = np.clip(np.asarray(anomaly_map, dtype=np.float32), 0, 1)
    gt_mask = normalize_mask_to_uint8(gt_mask)

    if resize_mask is not None:
        if isinstance(resize_mask, int):
            target_h, target_w = resize_mask, resize_mask
        else:
            target_h, target_w = resize_mask

        anomaly_map = cv2.resize(anomaly_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        gt_mask = cv2.resize(gt_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        gt_mask = normalize_mask_to_uint8(gt_mask)

    return anomaly_map, gt_mask


def append_merged_record_to_eval_lists(
    merged_record,
    gt_list_px,
    pr_list_px,
    gt_list_sp,
    pr_list_sp,
    anomaly_map_list,
    gt_mask_list,
    max_ratio=0.01,
    resize_mask=None,
):
    anomaly_map, gt_mask = resize_merged_record_for_eval(
        merged_record["anomaly_map"],
        merged_record["gt_mask"],
        resize_mask=resize_mask,
    )

    gt_binary = (gt_mask > 127).astype(np.float32)

    anomaly_map_list.append(np.clip(anomaly_map * 255.0, 0, 255).astype(np.uint8))
    gt_mask_list.append((gt_binary * 255).astype(np.uint8))

    pr_list_px.append(anomaly_map.astype(np.float32))
    gt_list_px.append(gt_binary.astype(np.float32))

    gt_list_sp.append(float(np.any(gt_binary > 0.5)))

    anomaly_map_flat = anomaly_map.reshape(-1)

    if max_ratio == 0:
        sp_score = np.max(anomaly_map_flat)
    else:
        k = max(1, int(anomaly_map_flat.shape[0] * max_ratio))
        sp_score = np.partition(anomaly_map_flat, -k)[-k:].mean()

    pr_list_sp.append(float(sp_score))


def init_score_stats(hist_bins=256):
    return {
        "sum": 0.0,
        "sumsq": 0.0,
        "count": 0,
        "hist_bins": int(hist_bins),
        "hist": np.zeros(int(hist_bins), dtype=np.int64),
    }


def update_running_stats(stats, anomaly_map, resize_mask=None):
    anomaly_map = np.clip(np.asarray(anomaly_map, dtype=np.float32), 0, 1)

    if resize_mask is not None:
        if isinstance(resize_mask, int):
            target_h, target_w = resize_mask, resize_mask
        else:
            target_h, target_w = resize_mask
        anomaly_map = cv2.resize(anomaly_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        anomaly_map = np.clip(anomaly_map, 0, 1)

    values = anomaly_map.astype(np.float64).reshape(-1)
    stats["sum"] += float(values.sum())
    stats["sumsq"] += float(np.square(values).sum())
    stats["count"] += int(values.size)

    if "hist" in stats:
        hist_bins = int(stats.get("hist_bins", len(stats["hist"])))
        value_bins = np.clip((values * (hist_bins - 1)).astype(np.int64), 0, hist_bins - 1)
        stats["hist"] += np.bincount(value_bins, minlength=hist_bins).astype(np.int64)


def merge_stats(dst, src):
    dst["sum"] += src["sum"]
    dst["sumsq"] += src["sumsq"]
    dst["count"] += src["count"]

    if "hist" in dst and "hist" in src:
        if int(dst.get("hist_bins", len(dst["hist"]))) != int(src.get("hist_bins", len(src["hist"]))):
            raise ValueError("hist_bins 不一致，无法合并统计量。")
        dst["hist"] += src["hist"]


def get_mean_std_from_stats(stats):
    if stats["count"] == 0:
        raise ValueError("stats count 为 0，无法计算均值和标准差。")

    mean_value = stats["sum"] / stats["count"]
    var_value = stats["sumsq"] / stats["count"] - mean_value ** 2
    var_value = max(var_value, 0.0)
    std_value = np.sqrt(var_value)

    return mean_value, std_value


def get_tail_prob_from_hist(stats, threshold):
    if "hist" not in stats or stats["count"] == 0:
        raise ValueError("需要 histogram 才能计算 tail probability。")

    hist = stats["hist"]
    hist_bins = int(stats.get("hist_bins", len(hist)))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    threshold_bin = int(np.ceil(threshold * (hist_bins - 1)))
    threshold_bin = int(np.clip(threshold_bin, 0, hist_bins - 1))

    return float(hist[threshold_bin:].sum() / max(int(stats["count"]), 1))


def load_dtd_image_paths(dtd_path):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_paths = []

    if dtd_path is None or len(str(dtd_path)) == 0:
        return image_paths

    if os.path.isfile(dtd_path):
        ext = os.path.splitext(dtd_path)[1].lower()
        if ext in image_extensions:
            image_paths.append(dtd_path)
        return image_paths

    if not os.path.isdir(dtd_path):
        return image_paths

    for root, _, files in os.walk(dtd_path):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_paths.append(os.path.join(root, filename))

    image_paths.sort()
    return image_paths


def make_synthetic_img_path(img_path, repeat_idx):
    img_dir = os.path.dirname(img_path)
    ext = os.path.splitext(img_path)[1]
    base_name, coord = parse_crop_coordinate_from_path(img_path)

    if coord is None:
        filename = f"{base_name}_syn{repeat_idx:03d}{ext}"
    else:
        x1, y1, x2, y2 = coord
        filename = f"{base_name}_syn{repeat_idx:03d}_x{x1}_y{y1}_x{x2}_y{y2}{ext}"

    return os.path.join(img_dir, filename)


def make_illumination_img_path(img_path, repeat_idx):
    img_dir = os.path.dirname(img_path)
    ext = os.path.splitext(img_path)[1]
    base_name, coord = parse_crop_coordinate_from_path(img_path)

    if coord is None:
        filename = f"{base_name}_illum{repeat_idx:03d}{ext}"
    else:
        x1, y1, x2, y2 = coord
        filename = f"{base_name}_illum{repeat_idx:03d}_x{x1}_y{y1}_x{x2}_y{y2}{ext}"

    return os.path.join(img_dir, filename)


def _get_odd_kernel_size(value, max_size):
    value = int(max(3, value))
    if value % 2 == 0:
        value += 1

    max_size = int(max(3, max_size))
    if max_size % 2 == 0:
        max_size -= 1

    return int(max(3, min(value, max_size)))


def apply_random_local_spot_shadow_to_img01(
    img01,
    rng,
    region_num_min=1,
    region_num_max=3,
    area_min=0.005,
    area_max=0.08,
    bright_strength_min=0.15,
    bright_strength_max=0.50,
    shadow_strength_min=0.10,
    shadow_strength_max=0.40,
    shadow_prob=0.5,
    noise_prob=0.5,
    noise_strength=0.25,
    blur_ratio_min=0.35,
    blur_ratio_max=0.85,
):
    if img01.ndim != 3:
        return img01

    device = img01.device
    dtype = img01.dtype
    _, h, w = img01.shape

    if h <= 1 or w <= 1:
        return img01

    region_num_min = max(1, int(region_num_min))
    region_num_max = max(region_num_min, int(region_num_max))
    num_regions = int(rng.randint(region_num_min, region_num_max + 1))

    area_min = float(np.clip(area_min, 1.0 / max(h * w, 1), 1.0))
    area_max = float(np.clip(area_max, area_min, 1.0))
    bright_strength_min = max(float(bright_strength_min), 0.0)
    bright_strength_max = max(float(bright_strength_max), bright_strength_min)
    shadow_strength_min = max(float(shadow_strength_min), 0.0)
    shadow_strength_max = max(float(shadow_strength_max), shadow_strength_min)
    shadow_prob = float(np.clip(shadow_prob, 0.0, 1.0))
    noise_prob = float(np.clip(noise_prob, 0.0, 1.0))
    noise_strength = max(float(noise_strength), 0.0)
    blur_ratio_min = max(float(blur_ratio_min), 0.0)
    blur_ratio_max = max(float(blur_ratio_max), blur_ratio_min)

    for _ in range(num_regions):
        area_ratio = float(rng.uniform(area_min, area_max))
        target_area = max(area_ratio * h * w, 1.0)
        aspect_ratio = float(np.exp(rng.uniform(np.log(0.35), np.log(2.85))))

        rx = int(max(2, math.sqrt(target_area * aspect_ratio / math.pi)))
        ry = int(max(2, math.sqrt(target_area / (math.pi * aspect_ratio))))
        rx = int(min(rx, max(2, w)))
        ry = int(min(ry, max(2, h)))

        center_x = int(rng.randint(0, max(w, 1)))
        center_y = int(rng.randint(0, max(h, 1)))
        angle = float(rng.uniform(0.0, 180.0))

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(
            mask,
            (center_x, center_y),
            (rx, ry),
            angle,
            0,
            360,
            1.0,
            thickness=-1,
        )

        blur_ratio = float(rng.uniform(blur_ratio_min, blur_ratio_max))
        blur_size = _get_odd_kernel_size(max(rx, ry) * blur_ratio * 2.0 + 1.0, max(h, w) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        max_value = float(mask.max())
        if max_value <= 1e-8:
            continue
        mask = mask / max_value

        if rng.rand() < noise_prob and noise_strength > 0:
            grid_h = int(rng.randint(3, 7))
            grid_w = int(rng.randint(3, 7))
            low_res_field = rng.uniform(-1.0, 1.0, size=(grid_h, grid_w)).astype(np.float32)
            field = cv2.resize(low_res_field, (w, h), interpolation=cv2.INTER_CUBIC)
            max_abs = float(np.max(np.abs(field)))
            if max_abs > 1e-8:
                field = field / max_abs
            mask = mask * np.clip(1.0 + noise_strength * field, 0.0, None)
            mask = np.clip(mask, 0.0, 1.0)

        mask_tensor = torch.from_numpy(mask).to(device=device, dtype=dtype).view(1, h, w)

        if rng.rand() < shadow_prob:
            strength = float(rng.uniform(shadow_strength_min, shadow_strength_max))
            local_factor = torch.clamp(1.0 - strength * mask_tensor, min=0.05)
        else:
            strength = float(rng.uniform(bright_strength_min, bright_strength_max))
            local_factor = 1.0 + strength * mask_tensor

        img01 = torch.clamp(img01 * local_factor, 0.0, 1.0)

    return img01


def apply_random_global_illumination_to_img01(
    img01,
    rng,
    exposure_min=0.45,
    exposure_max=1.8,
    gamma_min=0.6,
    gamma_max=1.8,
    contrast_min=0.7,
    contrast_max=1.5,
    channel_gain_min=0.85,
    channel_gain_max=1.15,
    local_prob=0.5,
    local_strength_min=0.0,
    local_strength_max=0.35,
):
    device = img01.device
    dtype = img01.dtype
    c, h, w = img01.shape

    exposure_min = max(float(exposure_min), 1e-6)
    exposure_max = max(float(exposure_max), exposure_min)
    gamma_min = max(float(gamma_min), 1e-6)
    gamma_max = max(float(gamma_max), gamma_min)
    contrast_min = max(float(contrast_min), 1e-6)
    contrast_max = max(float(contrast_max), contrast_min)
    channel_gain_min = max(float(channel_gain_min), 1e-6)
    channel_gain_max = max(float(channel_gain_max), channel_gain_min)

    if exposure_min < 1.0 < exposure_max:
        if rng.rand() < 0.5:
            exposure = float(rng.uniform(exposure_min, 1.0))
        else:
            exposure = float(rng.uniform(1.0, exposure_max))
    else:
        exposure = float(rng.uniform(exposure_min, exposure_max))

    gamma = float(rng.uniform(gamma_min, gamma_max))
    contrast = float(rng.uniform(contrast_min, contrast_max))

    img01 = torch.clamp(img01 * exposure, 0.0, 1.0)
    img01 = torch.clamp(torch.pow(torch.clamp(img01, 1e-6, 1.0), gamma), 0.0, 1.0)
    img01 = torch.clamp((img01 - 0.5) * contrast + 0.5, 0.0, 1.0)

    if c >= 3:
        gains = torch.ones((c, 1, 1), device=device, dtype=dtype)
        rgb_gains = rng.uniform(channel_gain_min, channel_gain_max, size=3).astype(np.float32)
        gains[:3, 0, 0] = torch.tensor(rgb_gains, device=device, dtype=dtype)
        img01 = torch.clamp(img01 * gains, 0.0, 1.0)

    if rng.rand() < float(np.clip(local_prob, 0.0, 1.0)):
        local_strength_min = max(float(local_strength_min), 0.0)
        local_strength_max = max(float(local_strength_max), local_strength_min)
        local_strength = float(rng.uniform(local_strength_min, local_strength_max))

        if local_strength > 0:
            grid_h = int(rng.randint(3, 7))
            grid_w = int(rng.randint(3, 7))
            low_res_field = rng.uniform(-1.0, 1.0, size=(grid_h, grid_w)).astype(np.float32)
            field = cv2.resize(low_res_field, (w, h), interpolation=cv2.INTER_CUBIC)
            max_abs = float(np.max(np.abs(field)))
            if max_abs > 1e-8:
                field = field / max_abs
            field_tensor = torch.from_numpy(field).to(device=device, dtype=dtype).view(1, h, w)
            local_factor = torch.clamp(1.0 + local_strength * field_tensor, min=0.05)
            img01 = torch.clamp(img01 * local_factor, 0.0, 1.0)

    return img01


def apply_random_illumination_to_tensor(
    img,
    rng,
    exposure_min=0.45,
    exposure_max=1.8,
    gamma_min=0.6,
    gamma_max=1.8,
    contrast_min=0.7,
    contrast_max=1.5,
    channel_gain_min=0.85,
    channel_gain_max=1.15,
    local_prob=0.5,
    local_strength_min=0.0,
    local_strength_max=0.35,
    local_spot_prob=0.7,
    local_region_num_min=1,
    local_region_num_max=3,
    local_area_min=0.005,
    local_area_max=0.08,
    local_bright_strength_min=0.15,
    local_bright_strength_max=0.50,
    local_shadow_strength_min=0.10,
    local_shadow_strength_max=0.40,
    local_shadow_prob=0.5,
    local_noise_prob=0.5,
    local_noise_strength=0.25,
    local_blur_ratio_min=0.35,
    local_blur_ratio_max=0.85,
):
    if not torch.is_tensor(img):
        raise TypeError("illumination augmentation 需要输入 torch.Tensor。")

    if img.ndim != 3:
        return img

    device = img.device
    dtype = img.dtype
    c, h, w = img.shape

    img_min = float(img.detach().min().cpu())
    img_max = float(img.detach().max().cpu())

    use_imagenet_denorm = c == 3 and (img_min < -0.05 or img_max > 1.05)

    if use_imagenet_denorm:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(3, 1, 1)
        img01 = torch.clamp(img * std + mean, 0.0, 1.0)
    else:
        mean = None
        std = None
        img01 = torch.clamp(img, 0.0, 1.0)

    if rng.rand() < float(np.clip(local_spot_prob, 0.0, 1.0)):
        img01 = apply_random_local_spot_shadow_to_img01(
            img01=img01,
            rng=rng,
            region_num_min=local_region_num_min,
            region_num_max=local_region_num_max,
            area_min=local_area_min,
            area_max=local_area_max,
            bright_strength_min=local_bright_strength_min,
            bright_strength_max=local_bright_strength_max,
            shadow_strength_min=local_shadow_strength_min,
            shadow_strength_max=local_shadow_strength_max,
            shadow_prob=local_shadow_prob,
            noise_prob=local_noise_prob,
            noise_strength=local_noise_strength,
            blur_ratio_min=local_blur_ratio_min,
            blur_ratio_max=local_blur_ratio_max,
        )
    else:
        img01 = apply_random_global_illumination_to_img01(
            img01=img01,
            rng=rng,
            exposure_min=exposure_min,
            exposure_max=exposure_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            contrast_min=contrast_min,
            contrast_max=contrast_max,
            channel_gain_min=channel_gain_min,
            channel_gain_max=channel_gain_max,
            local_prob=local_prob,
            local_strength_min=local_strength_min,
            local_strength_max=local_strength_max,
        )

    if use_imagenet_denorm:
        img_aug = (img01 - mean) / std
    else:
        img_aug = img01

    return img_aug.to(dtype=dtype)


class IlluminationAugmentedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        repeat=1,
        seed=321,
        exposure_min=0.45,
        exposure_max=1.8,
        gamma_min=0.6,
        gamma_max=1.8,
        contrast_min=0.7,
        contrast_max=1.5,
        channel_gain_min=0.85,
        channel_gain_max=1.15,
        local_prob=0.5,
        local_strength_min=0.0,
        local_strength_max=0.35,
        illumination_prob=0.5,
        local_spot_prob=0.7,
        local_region_num_min=1,
        local_region_num_max=3,
        local_area_min=0.005,
        local_area_max=0.08,
        local_bright_strength_min=0.15,
        local_bright_strength_max=0.50,
        local_shadow_strength_min=0.10,
        local_shadow_strength_max=0.40,
        local_shadow_prob=0.5,
        local_noise_prob=0.5,
        local_noise_strength=0.25,
        local_blur_ratio_min=0.35,
        local_blur_ratio_max=0.85,
    ):
        self.base_dataset = base_dataset
        self.repeat = max(1, int(repeat))
        self.seed = int(seed)
        self.exposure_min = float(exposure_min)
        self.exposure_max = float(exposure_max)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.contrast_min = float(contrast_min)
        self.contrast_max = float(contrast_max)
        self.channel_gain_min = float(channel_gain_min)
        self.channel_gain_max = float(channel_gain_max)
        self.local_prob = float(local_prob)
        self.local_strength_min = float(local_strength_min)
        self.local_strength_max = float(local_strength_max)
        self.illumination_prob = float(np.clip(illumination_prob, 0.0, 1.0))
        self.local_spot_prob = float(np.clip(local_spot_prob, 0.0, 1.0))
        self.local_region_num_min = int(local_region_num_min)
        self.local_region_num_max = int(local_region_num_max)
        self.local_area_min = float(local_area_min)
        self.local_area_max = float(local_area_max)
        self.local_bright_strength_min = float(local_bright_strength_min)
        self.local_bright_strength_max = float(local_bright_strength_max)
        self.local_shadow_strength_min = float(local_shadow_strength_min)
        self.local_shadow_strength_max = float(local_shadow_strength_max)
        self.local_shadow_prob = float(np.clip(local_shadow_prob, 0.0, 1.0))
        self.local_noise_prob = float(np.clip(local_noise_prob, 0.0, 1.0))
        self.local_noise_strength = float(local_noise_strength)
        self.local_blur_ratio_min = float(local_blur_ratio_min)
        self.local_blur_ratio_max = float(local_blur_ratio_max)

        base_image_paths = extract_image_paths_from_dataset(base_dataset)
        self.base_image_paths = base_image_paths

        if base_image_paths is not None:
            self.img_paths = []
            for repeat_idx in range(self.repeat):
                for img_path in base_image_paths:
                    self.img_paths.append(make_illumination_img_path(img_path, repeat_idx))

    def __len__(self):
        return len(self.base_dataset) * self.repeat

    def __getitem__(self, idx):
        base_len = len(self.base_dataset)
        repeat_idx = idx // base_len
        base_idx = idx % base_len

        img, gt, label, img_path = self.base_dataset[base_idx]

        rng = np.random.RandomState(self.seed + idx * 104729 + repeat_idx * 9973)
        apply_illumination = rng.rand() < self.illumination_prob

        if apply_illumination:
            img_aug = apply_random_illumination_to_tensor(
                img=img.clone(),
                rng=rng,
                exposure_min=self.exposure_min,
                exposure_max=self.exposure_max,
                gamma_min=self.gamma_min,
                gamma_max=self.gamma_max,
                contrast_min=self.contrast_min,
                contrast_max=self.contrast_max,
                channel_gain_min=self.channel_gain_min,
                channel_gain_max=self.channel_gain_max,
                local_prob=self.local_prob,
                local_strength_min=self.local_strength_min,
                local_strength_max=self.local_strength_max,
                local_spot_prob=self.local_spot_prob,
                local_region_num_min=self.local_region_num_min,
                local_region_num_max=self.local_region_num_max,
                local_area_min=self.local_area_min,
                local_area_max=self.local_area_max,
                local_bright_strength_min=self.local_bright_strength_min,
                local_bright_strength_max=self.local_bright_strength_max,
                local_shadow_strength_min=self.local_shadow_strength_min,
                local_shadow_strength_max=self.local_shadow_strength_max,
                local_shadow_prob=self.local_shadow_prob,
                local_noise_prob=self.local_noise_prob,
                local_noise_strength=self.local_noise_strength,
                local_blur_ratio_min=self.local_blur_ratio_min,
                local_blur_ratio_max=self.local_blur_ratio_max,
            )
        else:
            img_aug = img

        illum_path = make_illumination_img_path(img_path, repeat_idx)

        return img_aug, gt, label, illum_path


def generate_synthetic_blob_masks(
    h,
    w,
    rng,
    area_min=0.001,
    area_max=0.10,
    soft_blur_prob=0.7,
):
    area_min = float(np.clip(area_min, 1.0 / max(h * w, 1), 1.0))
    area_max = float(np.clip(area_max, area_min, 1.0))
    target_area_ratio = float(rng.uniform(area_min, area_max))

    best_mask = None
    best_area_error = float("inf")

    for _ in range(50):
        mask = np.zeros((h, w), dtype=np.uint8)
        num_blobs = int(rng.randint(1, 4))
        area_per_blob = max(target_area_ratio * h * w / num_blobs, 1.0)

        for _ in range(num_blobs):
            center_x = int(rng.randint(0, max(w, 1)))
            center_y = int(rng.randint(0, max(h, 1)))
            radius = math.sqrt(area_per_blob / math.pi)

            rx = int(max(2, radius * rng.uniform(0.6, 2.0)))
            ry = int(max(2, radius * rng.uniform(0.6, 2.0)))
            rx = int(min(rx, max(w // 2, 2)))
            ry = int(min(ry, max(h // 2, 2)))
            angle = float(rng.uniform(0, 180))

            cv2.ellipse(
                mask,
                (center_x, center_y),
                (rx, ry),
                angle,
                0,
                360,
                1,
                thickness=-1,
            )

        area_ratio = float(mask.mean())
        area_error = abs(area_ratio - target_area_ratio)

        if area_ratio > 0 and area_error < best_area_error:
            best_mask = mask.copy()
            best_area_error = area_error

        if area_ratio >= area_min * 0.5 and area_ratio <= area_max * 1.5:
            best_mask = mask.copy()
            break

    if best_mask is None or best_mask.max() == 0:
        best_mask = np.zeros((h, w), dtype=np.uint8)
        center_x = int(w // 2)
        center_y = int(h // 2)
        radius = max(2, int(math.sqrt(area_min * h * w / math.pi)))
        cv2.circle(best_mask, (center_x, center_y), radius, 1, thickness=-1)

    gt_mask = best_mask.astype(np.float32)
    soft_mask = gt_mask.copy()

    if rng.rand() < soft_blur_prob:
        blur_ksize = int(rng.choice([5, 7, 9, 11, 15]))
        blur_ksize = min(blur_ksize, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
        if blur_ksize >= 3:
            soft_mask = cv2.GaussianBlur(soft_mask, (blur_ksize, blur_ksize), 0)
            max_value = float(soft_mask.max())
            if max_value > 0:
                soft_mask = soft_mask / max_value

    soft_mask = np.clip(soft_mask, 0, 1).astype(np.float32)
    gt_mask = (gt_mask > 0.5).astype(np.float32)

    return gt_mask, soft_mask


class DTDSyntheticAnomalyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        dtd_image_paths,
        dtd_transform,
        repeat=1,
        seed=1,
        area_min=0.001,
        area_max=0.10,
        alpha_min=0.4,
        alpha_max=0.9,
        soft_blur_prob=0.7,
    ):
        self.base_dataset = base_dataset
        self.dtd_image_paths = list(dtd_image_paths)
        self.dtd_transform = dtd_transform
        self.repeat = max(1, int(repeat))
        self.seed = int(seed)
        self.area_min = float(area_min)
        self.area_max = float(area_max)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.soft_blur_prob = float(soft_blur_prob)

        if len(self.dtd_image_paths) == 0:
            raise ValueError("DTD 图像路径为空，无法生成合成异常。")

        base_image_paths = extract_image_paths_from_dataset(base_dataset)
        self.base_image_paths = base_image_paths

        if base_image_paths is not None:
            self.img_paths = []
            for repeat_idx in range(self.repeat):
                for img_path in base_image_paths:
                    self.img_paths.append(make_synthetic_img_path(img_path, repeat_idx))

    def __len__(self):
        return len(self.base_dataset) * self.repeat

    def __getitem__(self, idx):
        base_len = len(self.base_dataset)
        repeat_idx = idx // base_len
        base_idx = idx % base_len

        img, gt, label, img_path = self.base_dataset[base_idx]

        rng = np.random.RandomState(self.seed + idx * 9973 + repeat_idx * 101)
        dtd_idx = int(rng.randint(0, len(self.dtd_image_paths)))
        dtd_path = self.dtd_image_paths[dtd_idx]

        with Image.open(dtd_path) as texture_image:
            texture_image = texture_image.convert("RGB")
            texture = self.dtd_transform(texture_image)

        if not torch.is_tensor(texture):
            raise TypeError("dtd_transform 必须返回 torch.Tensor。")

        img = img.clone()
        texture = texture.to(dtype=img.dtype)

        if texture.ndim == 2:
            texture = texture.unsqueeze(0)

        if texture.shape[0] == 1 and img.shape[0] == 3:
            texture = texture.repeat(3, 1, 1)

        h, w = img.shape[-2], img.shape[-1]

        if texture.shape[-2:] != (h, w):
            texture = F.interpolate(
                texture.unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        gt_mask_np, soft_mask_np = generate_synthetic_blob_masks(
            h=h,
            w=w,
            rng=rng,
            area_min=self.area_min,
            area_max=self.area_max,
            soft_blur_prob=self.soft_blur_prob,
        )

        alpha = float(rng.uniform(self.alpha_min, self.alpha_max))
        soft_mask = torch.from_numpy(soft_mask_np).to(dtype=img.dtype)
        gt_mask = torch.from_numpy(gt_mask_np).to(dtype=img.dtype)

        blend_mask = soft_mask.unsqueeze(0) * alpha
        syn_img = img * (1.0 - blend_mask) + texture * blend_mask
        syn_gt = gt_mask.unsqueeze(0)

        syn_path = make_synthetic_img_path(img_path, repeat_idx)

        return syn_img, syn_gt, torch.tensor(1, dtype=torch.long), syn_path


def build_illumination_augmented_dataset(base_dataset, args):
    return IlluminationAugmentedDataset(
        base_dataset=base_dataset,
        repeat=args.illumination_repeat,
        seed=args.illumination_seed,
        exposure_min=args.illumination_exposure_min,
        exposure_max=args.illumination_exposure_max,
        gamma_min=args.illumination_gamma_min,
        gamma_max=args.illumination_gamma_max,
        contrast_min=args.illumination_contrast_min,
        contrast_max=args.illumination_contrast_max,
        channel_gain_min=args.illumination_channel_gain_min,
        channel_gain_max=args.illumination_channel_gain_max,
        local_prob=args.illumination_local_prob,
        local_strength_min=args.illumination_local_strength_min,
        local_strength_max=args.illumination_local_strength_max,
        illumination_prob=args.illumination_prob,
        local_spot_prob=args.illumination_local_spot_prob,
        local_region_num_min=args.illumination_local_region_num_min,
        local_region_num_max=args.illumination_local_region_num_max,
        local_area_min=args.illumination_local_area_min,
        local_area_max=args.illumination_local_area_max,
        local_bright_strength_min=args.illumination_local_bright_strength_min,
        local_bright_strength_max=args.illumination_local_bright_strength_max,
        local_shadow_strength_min=args.illumination_local_shadow_strength_min,
        local_shadow_strength_max=args.illumination_local_shadow_strength_max,
        local_shadow_prob=args.illumination_local_shadow_prob,
        local_noise_prob=args.illumination_local_noise_prob,
        local_noise_strength=args.illumination_local_noise_strength,
        local_blur_ratio_min=args.illumination_local_blur_ratio_min,
        local_blur_ratio_max=args.illumination_local_blur_ratio_max,
    )


def collect_merged_maps_for_threshold_search(
    model,
    dataloader,
    device,
    resize_mask=256,
    merge_mode="average",
    normalize_amap=False,
):
    model.eval()

    anomaly_map_list = []
    gt_mask_list = []

    current_record = None
    current_key = None

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    def append_record(record):
        merged_record = finalize_single_merge_record(record, merge_mode=merge_mode)
        anomaly_map, gt_mask = resize_merged_record_for_eval(
            merged_record["anomaly_map"],
            merged_record["gt_mask"],
            resize_mask=resize_mask,
        )
        anomaly_map_list.append(anomaly_map.astype(np.float32))
        gt_mask_list.append((gt_mask > 127).astype(np.uint8))
        del merged_record

    with torch.no_grad():
        for img, gt, label, img_path_batch in tqdm(dataloader, ncols=80):
            img = img.to(device)
            output = model(img, return_seg=True, seg_out_size=img.shape[-2:])
            en, de, seg_logits = output[0], output[1], output[-1]

            anomaly_map_batch, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map_batch = compose_anomaly_map_with_seg(anomaly_map_batch, seg_logits, model.eval_seg_weight)
            anomaly_map_batch = gaussian_kernel(anomaly_map_batch)

            if normalize_amap:
                scores = []
                for idx in range(anomaly_map_batch.shape[0]):
                    anomaly_map = anomaly_map_batch[idx, 0].detach().cpu().numpy()
                    scores.append(anomaly_map)

                scores = np.array(scores)
                min_scores = np.min(scores)
                max_scores = np.max(scores)
                anomaly_scores = (scores - min_scores) / (max_scores - min_scores + 1e-10)
                anomaly_scores = np.clip(anomaly_scores, 0, 1)
            else:
                anomaly_scores = []
                for idx in range(anomaly_map_batch.shape[0]):
                    anomaly_map = anomaly_map_batch[idx, 0].detach().cpu().numpy()
                    anomaly_map = np.clip(anomaly_map, 0, 1)
                    anomaly_scores.append(anomaly_map)
                anomaly_scores = np.array(anomaly_scores)

            for idx, img_path in enumerate(img_path_batch):
                merge_key = get_merge_key_from_path(img_path)

                if current_key is not None and merge_key != current_key:
                    append_record(current_record)
                    current_record = None
                    current_key = None
                    gc.collect()

                valid_h, valid_w = get_valid_crop_size_from_path(img_path, anomaly_scores[idx].shape)
                anomaly_map = cv2.resize(
                    anomaly_scores[idx].astype(np.float32),
                    (valid_w, valid_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                gt_mask = get_gt_mask_from_tensor(gt, idx, valid_h, valid_w)

                current_record = add_crop_to_merge_record(
                    record=current_record,
                    img_path=img_path,
                    anomaly_map=anomaly_map,
                    gt_mask=gt_mask,
                    merge_mode=merge_mode,
                )
                current_key = merge_key

            del img, gt, output, en, de, seg_logits, anomaly_map_batch, anomaly_scores

    if current_record is not None:
        append_record(current_record)
        del current_record
        gc.collect()

    return anomaly_map_list, gt_mask_list


def search_best_k_from_synthetic(
    anomaly_map_list,
    gt_mask_list,
    mean_01,
    std_01,
    base_k=5.0,
    min_k=3.0,
    max_k=8.0,
    step=0.25,
    normal_stats=None,
    normal_fp_weight=0.0,
    k_reg_weight=0.0,
):
    if len(anomaly_map_list) == 0 or len(gt_mask_list) == 0:
        return {
            "synthetic_best_k": float(base_k),
            "synthetic_best_threshold_01": float(np.clip(mean_01 + base_k * std_01, 0.0, 1.0)),
            "synthetic_best_threshold_255": float(np.clip(mean_01 + base_k * std_01, 0.0, 1.0) * 255.0),
            "synthetic_best_f1": 0.0,
            "synthetic_best_iou": 0.0,
            "synthetic_best_precision": 0.0,
            "synthetic_best_recall": 0.0,
            "synthetic_best_score": 0.0,
            "synthetic_best_normal_tail_prob": None,
        }

    if std_01 <= 1e-12:
        candidate_ks = np.array([base_k], dtype=np.float32)
    else:
        candidate_ks = np.arange(min_k, max_k + 1e-9, step, dtype=np.float32)

    best_result = None

    for k in candidate_ks:
        threshold_01 = float(np.clip(mean_01 + float(k) * std_01, 0.0, 1.0))

        tp = 0.0
        fp = 0.0
        fn = 0.0

        for anomaly_map, gt_mask in zip(anomaly_map_list, gt_mask_list):
            pred = anomaly_map >= threshold_01
            gt = gt_mask > 0

            tp += float(np.logical_and(pred, gt).sum())
            fp += float(np.logical_and(pred, np.logical_not(gt)).sum())
            fn += float(np.logical_and(np.logical_not(pred), gt).sum())

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        if normal_stats is not None:
            normal_tail_prob = get_tail_prob_from_hist(normal_stats, threshold_01)
        else:
            normal_tail_prob = 0.0

        score = f1
        score -= float(normal_fp_weight) * float(normal_tail_prob)
        score -= float(k_reg_weight) * abs(float(k) - float(base_k))

        result = {
            "synthetic_best_k": float(k),
            "synthetic_best_threshold_01": float(threshold_01),
            "synthetic_best_threshold_255": float(threshold_01 * 255.0),
            "synthetic_best_f1": float(f1),
            "synthetic_best_iou": float(iou),
            "synthetic_best_precision": float(precision),
            "synthetic_best_recall": float(recall),
            "synthetic_best_score": float(score),
            "synthetic_best_normal_tail_prob": float(normal_tail_prob),
        }

        if best_result is None:
            best_result = result
        else:
            better_score = result["synthetic_best_score"] > best_result["synthetic_best_score"] + 1e-12
            same_score_better_f1 = (
                abs(result["synthetic_best_score"] - best_result["synthetic_best_score"]) <= 1e-12
                and result["synthetic_best_f1"] > best_result["synthetic_best_f1"] + 1e-12
            )
            same_f1_closer_base = (
                abs(result["synthetic_best_score"] - best_result["synthetic_best_score"]) <= 1e-12
                and abs(result["synthetic_best_f1"] - best_result["synthetic_best_f1"]) <= 1e-12
                and abs(result["synthetic_best_k"] - base_k) < abs(best_result["synthetic_best_k"] - base_k)
            )

            if better_score or same_score_better_f1 or same_f1_closer_base:
                best_result = result

    return best_result


def build_final_k_table(
    stats_by_item,
    synthetic_k_by_item=None,
    base_k=5.0,
    min_k=3.0,
    max_k=12.0,
):
    if not stats_by_item:
        raise ValueError("stats_by_item 为空，无法计算 final k。")

    synthetic_k_by_item = synthetic_k_by_item or {}
    final_table = []

    for item, stats in stats_by_item.items():
        mean_01, std_01 = get_mean_std_from_stats(stats)
        base_threshold_01 = float(np.clip(mean_01 + base_k * std_01, 0.0, 1.0))
        synthetic_result = synthetic_k_by_item.get(item, None)

        if synthetic_result is None:
            base_weight = 1.0
            synthetic_weight = 0.0
            final_k = float(base_k)
        else:
            synthetic_k = float(synthetic_result["synthetic_best_k"])
            base_weight = 0.5
            synthetic_weight = 0.5
            final_k = base_weight * float(base_k) + synthetic_weight * synthetic_k

        final_k = float(np.clip(final_k, min_k, max_k))
        final_threshold_01 = float(np.clip(mean_01 + final_k * std_01, 0.0, 1.0))

        row = {
            "item": item,
            "count": int(stats["count"]),
            "mean_01": float(mean_01),
            "std_01": float(std_01),
            "mean_255": float(mean_01 * 255.0),
            "std_255": float(std_01 * 255.0),
            "base_k": float(base_k),
            "base_threshold_01": float(base_threshold_01),
            "base_threshold_255": float(base_threshold_01 * 255.0),
            "base_weight": float(base_weight),
            "synthetic_weight": float(synthetic_weight),
            "final_k": float(final_k),
            "final_threshold_01": float(final_threshold_01),
            "final_threshold_255": float(final_threshold_01 * 255.0),
        }

        if synthetic_result is None:
            row["synthetic_best_k"] = None
            row["synthetic_best_threshold_01"] = None
            row["synthetic_best_threshold_255"] = None
            row["synthetic_best_f1"] = None
            row["synthetic_best_iou"] = None
            row["synthetic_best_precision"] = None
            row["synthetic_best_recall"] = None
            row["synthetic_best_score"] = None
            row["synthetic_best_normal_tail_prob"] = None
        else:
            for key, value in synthetic_result.items():
                row[key] = value

        final_table.append(row)

    return final_table



def make_eval_dataloader(dataset, args):
    dataset = SortedByMergeKeyDataset(dataset)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.eval_num_workers,
        pin_memory=args.pin_memory,
        collate_fn=four_tuple_collate,
    )

def build_merged_save_path(save_root, img_path, base_name, data_root, suffix=".tiff"):
    img_dir = os.path.dirname(img_path)
    relative_dir = os.path.relpath(img_dir, start=data_root)
    save_dir = os.path.join(save_root, relative_dir)
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, base_name + suffix)


def save_merged_anomaly_maps(
    model,
    dataloader,
    device,
    save_root=None,
    data_root=None,
    merge_mode="average",
    collect_anomaly_scores=False,
    normalize_amap=False,
    stats_resize_mask=None,
    save_maps=True,
):
    model.eval()

    current_record = None
    current_key = None
    stats = init_score_stats(hist_bins=256)

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    def finalize_save_and_release(record):
        merged_record = finalize_single_merge_record(record, merge_mode=merge_mode)
        anomaly_map = np.clip(merged_record["anomaly_map"], 0, 1)

        if save_maps:
            if save_root is None or data_root is None:
                raise ValueError("save_maps=True 时必须提供 save_root 和 data_root。")

            save_path = build_merged_save_path(
                save_root=save_root,
                img_path=merged_record["first_img_path"],
                base_name=merged_record["base_name"],
                data_root=data_root,
                suffix=".tiff",
            )

            anomaly_map_gray = np.clip(anomaly_map * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, anomaly_map_gray)
            del anomaly_map_gray

        if collect_anomaly_scores:
            update_running_stats(stats, anomaly_map, resize_mask=stats_resize_mask)

        del merged_record, anomaly_map

    with torch.no_grad():
        for img, gt, label, img_path_batch in tqdm(dataloader, ncols=80):
            img = img.to(device)
            output = model(img, return_seg=True, seg_out_size=img.shape[-2:])
            en, de, seg_logits = output[0], output[1], output[-1]

            anomaly_map_batch, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map_batch = compose_anomaly_map_with_seg(anomaly_map_batch, seg_logits, model.eval_seg_weight)
            anomaly_map_batch = gaussian_kernel(anomaly_map_batch)

            if normalize_amap:
                scores = []
                for idx in range(anomaly_map_batch.shape[0]):
                    anomaly_map = anomaly_map_batch[idx, 0].detach().cpu().numpy()
                    scores.append(anomaly_map)

                scores = np.array(scores)
                min_scores = np.min(scores)
                max_scores = np.max(scores)
                anomaly_scores = (scores - min_scores) / (max_scores - min_scores + 1e-10)
                anomaly_scores = np.clip(anomaly_scores, 0, 1)
            else:
                anomaly_scores = []
                for idx in range(anomaly_map_batch.shape[0]):
                    anomaly_map = anomaly_map_batch[idx, 0].detach().cpu().numpy()
                    anomaly_map = np.clip(anomaly_map, 0, 1)
                    anomaly_scores.append(anomaly_map)
                anomaly_scores = np.array(anomaly_scores)

            for idx, img_path in enumerate(img_path_batch):
                merge_key = get_merge_key_from_path(img_path)

                if current_key is not None and merge_key != current_key:
                    finalize_save_and_release(current_record)
                    current_record = None
                    current_key = None
                    gc.collect()

                valid_h, valid_w = get_valid_crop_size_from_path(img_path, anomaly_scores[idx].shape)
                anomaly_map = cv2.resize(
                    anomaly_scores[idx].astype(np.float32),
                    (valid_w, valid_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                gt_mask = get_gt_mask_from_tensor(gt, idx, valid_h, valid_w)

                current_record = add_crop_to_merge_record(
                    record=current_record,
                    img_path=img_path,
                    anomaly_map=anomaly_map,
                    gt_mask=gt_mask,
                    merge_mode=merge_mode,
                )
                current_key = merge_key

            del img, gt, output, en, de, seg_logits, anomaly_map_batch, anomaly_scores

    if current_record is not None:
        finalize_save_and_release(current_record)
        del current_record
        gc.collect()

    if collect_anomaly_scores:
        return stats

    return None


def load_trainable_state(trainable_modules, load_path, device):
    state_dict = torch.load(load_path, map_location=device)
    trainable_modules.load_state_dict(state_dict, strict=True)

def main(args):
    # Fixing the Random Seed
    setup_seed(1)

    def get_category_model_path(category):
        safe_category = str(category).replace(os.sep, "_")
        if os.altsep is not None:
            safe_category = safe_category.replace(os.altsep, "_")
        return os.path.join(args.save_dir, args.save_name, f"model_{safe_category}.pth")

    if is_dinov3_encoder(args.encoder):
        args.input_size = make_divisible_down(args.input_size, 16)
        args.crop_size = make_divisible_down(args.crop_size, 16)

    # Data Preparation
    data_transform, gt_transforms, data_transforms_train = get_data_transforms(args.input_size, args.crop_size)

    true_val_data_list = []
    for i, item in enumerate(args.item_list):
        true_val_path = os.path.join(args.data_path, item)

        true_val_data = MVTec2Dataset(root=true_val_path, transform=data_transform, phase="true_val", resize=args.input_size, normal_only=True)
        true_val_data_list.append(true_val_data)

    fuse_layer_encoder = [[0, 1], [2, 3], [4, 5], [6, 7]]
    fuse_layer_decoder = [[0, 1], [2, 3], [4, 5], [6, 7]]

    # Encoder info
    # encoder = vit_encoder.load(args.encoder)

    encoder = torch.hub.load(
        args.dinov3_repo,
        args.encoder,
        source='local',
        weights=args.dinov3_weights,
    )
    for p in encoder.parameters():
        p.requires_grad_(False)
    embed_dim, num_heads, target_layers = get_encoder_config(args.encoder)

    # Model Preparation
    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])

    INP = nn.ParameterList(
        [nn.Parameter(torch.randn(args.INP_num, embed_dim)) for _ in range(1)]
    )

    INP_Extractor = nn.ModuleList([
        Aggregation_Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
        )
        for _ in range(1)
    ])

    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
        )
        for _ in range(8)
    ])

    residual_head = ResidualSegHead([embed_dim] * len(fuse_layer_encoder), hidden_dim=args.seg_hidden_dim)

    model = INP_Former(
        encoder=encoder,
        bottleneck=Bottleneck,
        aggregation=INP_Extractor,
        decoder=INP_Guided_Decoder,
        target_layers=target_layers,
        remove_class_token=True,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        prototype_token=INP,
        residual_head=residual_head,
        eval_seg_weight=args.eval_seg_weight,
    )
    model = model.to(device)
    trainable_modules = nn.ModuleList([
        Bottleneck,
        INP_Guided_Decoder,
        INP_Extractor,
        INP,
        residual_head,
    ])

    if args.phase == 'true_val':

        all_stats = init_score_stats(hist_bins=256)
        stats_by_item = {}
        synthetic_k_by_item = {}

        illumination_all_stats = init_score_stats(hist_bins=256)
        illumination_stats_by_item = {}
        illumination_synthetic_k_by_item = {}

        dtd_image_paths = []
        if args.synthetic_k_enable:
            dtd_image_paths = load_dtd_image_paths(args.dtd_root)
            if len(dtd_image_paths) == 0:
                raise ValueError(f"没有在 DTD 路径中找到图像: {args.dtd_root}")
            print_fn(f"Loaded {len(dtd_image_paths)} DTD images from {args.dtd_root}")

        for item, true_val_data in zip(args.item_list, true_val_data_list):
            model_path = get_category_model_path(item)
            load_trainable_state(trainable_modules, model_path, device)
            model.eval()
            print_fn(f"Loaded trainable modules from {model_path}")

            true_val_dataloader = make_eval_dataloader(true_val_data, args)
            item_stats = save_merged_anomaly_maps(
                model=model,
                dataloader=true_val_dataloader,
                device=device,
                save_root=None,
                data_root=args.data_path,
                merge_mode=args.merge_mode,
                collect_anomaly_scores=True,
                normalize_amap=args.normalize_amap,
                stats_resize_mask=args.eval_resize_mask,
                save_maps=False,
            )
            stats_by_item[item] = item_stats
            merge_stats(all_stats, item_stats)

            del true_val_dataloader
            gc.collect()

            if args.synthetic_k_enable:
                mean_01, std_01 = get_mean_std_from_stats(item_stats)

                synthetic_dataset = DTDSyntheticAnomalyDataset(
                    base_dataset=true_val_data,
                    dtd_image_paths=dtd_image_paths,
                    dtd_transform=data_transform,
                    repeat=args.synthetic_repeat,
                    seed=args.synthetic_seed,
                    area_min=args.synthetic_area_min,
                    area_max=args.synthetic_area_max,
                    alpha_min=args.synthetic_alpha_min,
                    alpha_max=args.synthetic_alpha_max,
                    soft_blur_prob=args.synthetic_soft_blur_prob,
                )
                synthetic_dataloader = make_eval_dataloader(synthetic_dataset, args)

                synthetic_anomaly_maps, synthetic_gt_masks = collect_merged_maps_for_threshold_search(
                    model=model,
                    dataloader=synthetic_dataloader,
                    device=device,
                    resize_mask=args.eval_resize_mask,
                    merge_mode=args.merge_mode,
                    normalize_amap=args.normalize_amap,
                )

                synthetic_k_result = search_best_k_from_synthetic(
                    anomaly_map_list=synthetic_anomaly_maps,
                    gt_mask_list=synthetic_gt_masks,
                    mean_01=mean_01,
                    std_01=std_01,
                    base_k=args.base_k,
                    min_k=args.synthetic_k_min,
                    max_k=args.synthetic_k_max,
                    step=args.synthetic_k_step,
                    normal_stats=item_stats,
                    normal_fp_weight=args.synthetic_normal_fp_weight,
                    k_reg_weight=args.synthetic_k_reg_weight,
                )
                synthetic_k_by_item[item] = synthetic_k_result

                print_fn(
                    "{} synthetic best: k={:.4f}, threshold={:.2f}, F1={:.4f}, IoU={:.4f}, precision={:.4f}, recall={:.4f}, score={:.4f}".format(
                        item,
                        synthetic_k_result["synthetic_best_k"],
                        synthetic_k_result["synthetic_best_threshold_255"],
                        synthetic_k_result["synthetic_best_f1"],
                        synthetic_k_result["synthetic_best_iou"],
                        synthetic_k_result["synthetic_best_precision"],
                        synthetic_k_result["synthetic_best_recall"],
                        synthetic_k_result["synthetic_best_score"],
                    )
                )

                del synthetic_dataset, synthetic_dataloader, synthetic_anomaly_maps, synthetic_gt_masks
                gc.collect()

            if args.illumination_calibration_enable:
                illumination_true_val_data = build_illumination_augmented_dataset(true_val_data, args)
                illumination_true_val_dataloader = make_eval_dataloader(illumination_true_val_data, args)

                illumination_item_stats = save_merged_anomaly_maps(
                    model=model,
                    dataloader=illumination_true_val_dataloader,
                    device=device,
                    save_root=None,
                    data_root=args.data_path,
                    merge_mode=args.merge_mode,
                    collect_anomaly_scores=True,
                    normalize_amap=args.normalize_amap,
                    stats_resize_mask=args.eval_resize_mask,
                    save_maps=False,
                )
                illumination_stats_by_item[item] = illumination_item_stats
                merge_stats(illumination_all_stats, illumination_item_stats)

                del illumination_true_val_dataloader
                gc.collect()

                if args.synthetic_k_enable:
                    illumination_mean_01, illumination_std_01 = get_mean_std_from_stats(illumination_item_stats)

                    illumination_synthetic_dataset = DTDSyntheticAnomalyDataset(
                        base_dataset=illumination_true_val_data,
                        dtd_image_paths=dtd_image_paths,
                        dtd_transform=data_transform,
                        repeat=args.synthetic_repeat,
                        seed=args.synthetic_seed,
                        area_min=args.synthetic_area_min,
                        area_max=args.synthetic_area_max,
                        alpha_min=args.synthetic_alpha_min,
                        alpha_max=args.synthetic_alpha_max,
                        soft_blur_prob=args.synthetic_soft_blur_prob,
                    )
                    illumination_synthetic_dataloader = make_eval_dataloader(illumination_synthetic_dataset, args)

                    illumination_synthetic_anomaly_maps, illumination_synthetic_gt_masks = collect_merged_maps_for_threshold_search(
                        model=model,
                        dataloader=illumination_synthetic_dataloader,
                        device=device,
                        resize_mask=args.eval_resize_mask,
                        merge_mode=args.merge_mode,
                        normalize_amap=args.normalize_amap,
                    )

                    illumination_synthetic_k_result = search_best_k_from_synthetic(
                        anomaly_map_list=illumination_synthetic_anomaly_maps,
                        gt_mask_list=illumination_synthetic_gt_masks,
                        mean_01=illumination_mean_01,
                        std_01=illumination_std_01,
                        base_k=args.base_k,
                        min_k=args.synthetic_k_min,
                        max_k=args.synthetic_k_max,
                        step=args.synthetic_k_step,
                        normal_stats=illumination_item_stats,
                        normal_fp_weight=args.synthetic_normal_fp_weight,
                        k_reg_weight=args.synthetic_k_reg_weight,
                    )
                    illumination_synthetic_k_by_item[item] = illumination_synthetic_k_result

                    print_fn(
                        "{} illumination synthetic best: k={:.4f}, threshold={:.2f}, F1={:.4f}, IoU={:.4f}, precision={:.4f}, recall={:.4f}, score={:.4f}".format(
                            item,
                            illumination_synthetic_k_result["synthetic_best_k"],
                            illumination_synthetic_k_result["synthetic_best_threshold_255"],
                            illumination_synthetic_k_result["synthetic_best_f1"],
                            illumination_synthetic_k_result["synthetic_best_iou"],
                            illumination_synthetic_k_result["synthetic_best_precision"],
                            illumination_synthetic_k_result["synthetic_best_recall"],
                            illumination_synthetic_k_result["synthetic_best_score"],
                        )
                    )

                    del illumination_synthetic_dataset, illumination_synthetic_dataloader, illumination_synthetic_anomaly_maps, illumination_synthetic_gt_masks
                    gc.collect()

                del illumination_true_val_data
                gc.collect()

            current_final_row = build_final_k_table(
                stats_by_item={item: item_stats},
                synthetic_k_by_item={item: synthetic_k_by_item[item]} if item in synthetic_k_by_item else {},
                base_k=args.base_k,
                min_k=args.final_k_min,
                max_k=args.final_k_max,
            )[0]

            current_mixed_row = None
            if args.illumination_calibration_enable:
                current_mixed_row = build_final_k_table(
                    stats_by_item={item: illumination_stats_by_item[item]},
                    synthetic_k_by_item={item: illumination_synthetic_k_by_item[item]} if item in illumination_synthetic_k_by_item else {},
                    base_k=args.base_k,
                    min_k=args.final_k_min,
                    max_k=args.final_k_max,
                )[0]

            update_final_thresholds_by_split_json(
                category_name=item,
                normal_row=current_final_row,
                mixed_row=current_mixed_row,
                save_path=args.final_threshold_json_path,
                test_img_dirs=args.test_img_dirs,
                mixed_split_name="test_private_mixed",
                threshold_key="final_threshold_255",
            )

            print_fn(f"{item} thresholds updated to: {args.final_threshold_json_path}")

        if all_stats["count"] == 0:
            raise ValueError("没有收集到 true_val anomaly map，无法计算均值和标准差。")

        mean_value_01, std_value_01 = get_mean_std_from_stats(all_stats)
        mean_value = mean_value_01 * 255
        std_value = std_value_01 * 255

        print("所有 merge 后 anomaly map 像素的均值: ", mean_value)
        print("所有 merge 后 anomaly map 像素的标准差: ", std_value)

        final_k_table = build_final_k_table(
            stats_by_item=stats_by_item,
            synthetic_k_by_item=synthetic_k_by_item,
            base_k=args.base_k,
            min_k=args.final_k_min,
            max_k=args.final_k_max,
        )

        print_fn("Final k by category:")
        for row in final_k_table:
            print_fn(
                "{}: mean={:.4f}, std={:.4f}, base_k={:.4f}, k_syn={}, base_weight={:.4f}, synthetic_weight={:.4f}, final_k={:.4f}, final_threshold={:.2f}".format(
                    row["item"],
                    row["mean_255"],
                    row["std_255"],
                    row["base_k"],
                    "None" if row["synthetic_best_k"] is None else "{:.4f}".format(row["synthetic_best_k"]),
                    row["base_weight"],
                    row["synthetic_weight"],
                    row["final_k"],
                    row["final_threshold_255"],
                )
            )
        if args.illumination_calibration_enable:
            if illumination_all_stats["count"] == 0:
                raise ValueError("没有收集到 illumination true_val anomaly map，无法计算 mixed 均值和标准差。")

            illumination_mean_value_01, illumination_std_value_01 = get_mean_std_from_stats(illumination_all_stats)
            illumination_mean_value = illumination_mean_value_01 * 255
            illumination_std_value = illumination_std_value_01 * 255

            print("所有 illumination merge 后 anomaly map 像素的均值: ", illumination_mean_value)
            print("所有 illumination merge 后 anomaly map 像素的标准差: ", illumination_std_value)

            illumination_final_k_table = build_final_k_table(
                stats_by_item=illumination_stats_by_item,
                synthetic_k_by_item=illumination_synthetic_k_by_item,
                base_k=args.base_k,
                min_k=args.final_k_min,
                max_k=args.final_k_max,
            )

            print_fn("Illumination final k by category:")
            for row in illumination_final_k_table:
                print_fn(
                    "{}: mean={:.4f}, std={:.4f}, base_k={:.4f}, k_syn={}, base_weight={:.4f}, synthetic_weight={:.4f}, final_k={:.4f}, final_threshold={:.2f}".format(
                        row["item"],
                        row["mean_255"],
                        row["std_255"],
                        row["base_k"],
                        "None" if row["synthetic_best_k"] is None else "{:.4f}".format(row["synthetic_best_k"]),
                        row["base_weight"],
                        row["synthetic_weight"],
                        row["final_k"],
                        row["final_threshold_255"],
                    )
                )


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'Mvtec_ad_2') # 'MVTec-AD' or 'VisA' or 'Real-IAD'
    parser.add_argument('--data_path', type=str, default=r'./datasets/mvtec_ad_2_splits_1024') # Replace it with your path.
    parser.add_argument('--dtd_root', type=str, default='./datasets/dtd/images', help='Optional DTD root for Perlin + texture synthetic segmentation training')

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='INP-Former-Multi-Class')

    # model info
    parser.add_argument('--encoder', type=str, default='dinov3_vith16plus') # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--seg_hidden_dim', type=int, default=512)

    # training info
    parser.add_argument('--total_epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--phase', type=str, default='train') # true_val
    parser.add_argument('--test_img_dirs', nargs='+', default=['test_private', 'test_private_mixed'],)

    parser.add_argument('--eval_seg_weight', type=float, default=0.03)

    parser.add_argument('--merge_mode', type=str, default='average', choices=['average', 'max', 'overwrite'],
                        help='merge crop anomaly maps before metric/threshold/mean/std calculation')
    parser.add_argument('--eval_resize_mask', type=int, default=256,
                        help='先 merge 到原图空间，再 resize 到该尺寸后调用 ader_evaluator；设为小于等于0表示不 resize。')
    parser.add_argument('--normalize_amap', action='store_true',
                        help='是否沿用 evaluation_batch 中的 batch-level anomaly map normalization。')

    parser.add_argument('--base_k', type=float, default=5.0,
                        help='true_val 阶段用于 mean + k * std 的基础 k。')

    parser.add_argument('--synthetic_k_enable', action='store_true',
                        help='是否在 true_val 阶段使用 DTD 合成异常搜索 synthetic best k。')
    parser.add_argument('--synthetic_repeat', type=int, default=1,
                        help='每个 true_val crop 合成异常的重复次数。')
    parser.add_argument('--synthetic_seed', type=int, default=123,
                        help='DTD 合成异常随机种子。')
    parser.add_argument('--synthetic_area_min', type=float, default=0.001,
                        help='合成异常 mask 的最小面积比例。')
    parser.add_argument('--synthetic_area_max', type=float, default=0.10,
                        help='合成异常 mask 的最大面积比例。')
    parser.add_argument('--synthetic_alpha_min', type=float, default=0.4,
                        help='DTD 纹理混合 alpha 下限。')
    parser.add_argument('--synthetic_alpha_max', type=float, default=0.9,
                        help='DTD 纹理混合 alpha 上限。')
    parser.add_argument('--synthetic_soft_blur_prob', type=float, default=0.7,
                        help='合成异常 soft mask 使用 blur 的概率。')
    parser.add_argument('--synthetic_k_min', type=float, default=3.0,
                        help='synthetic best k 搜索下限。')
    parser.add_argument('--synthetic_k_max', type=float, default=15.0,
                        help='synthetic best k 搜索上限。')
    parser.add_argument('--synthetic_k_step', type=float, default=0.25,
                        help='synthetic best k 搜索步长。')
    parser.add_argument('--synthetic_normal_fp_weight', type=float, default=0.0,
                        help='synthetic k 搜索目标中 normal tail false positive penalty 权重。')
    parser.add_argument('--synthetic_k_reg_weight', type=float, default=0.0,
                        help='synthetic k 搜索目标中 |k-base_k| penalty 权重。')

    parser.add_argument('--final_k_min', type=float, default=3.0,
                        help='final_k 下限。')
    parser.add_argument('--final_k_max', type=float, default=12.0,
                        help='final_k 上限。')

    parser.add_argument('--illumination_calibration_enable', action='store_true',
                        help='是否额外构建曝光/欠曝光增强 true_val,用于 test_private_mixed 的阈值校准。')
    parser.add_argument('--illumination_repeat', type=int, default=1,
                        help='每个 true_val crop 生成 illumination 增强样本的重复次数。')
    parser.add_argument('--illumination_seed', type=int, default=321,
                        help='illumination 增强随机种子。')
    parser.add_argument('--illumination_prob', type=float, default=0.4,
                        help='对每个 illumination 样本应用光照模拟的概率；未命中时保留原图。')
    parser.add_argument('--illumination_local_spot_prob', type=float, default=0.7,
                        help='命中 illumination_prob 后，使用局部亮斑/阴影模拟而非全局光照模拟的概率。')
    parser.add_argument('--illumination_local_region_num_min', type=int, default=1,
                        help='局部亮斑/阴影区域数量下限。')
    parser.add_argument('--illumination_local_region_num_max', type=int, default=3,
                        help='局部亮斑/阴影区域数量上限。')
    parser.add_argument('--illumination_local_area_min', type=float, default=0.01,
                        help='单个局部亮斑/阴影区域面积占图像面积的下限。')
    parser.add_argument('--illumination_local_area_max', type=float, default=0.05,
                        help='单个局部亮斑/阴影区域面积占图像面积的上限。')
    parser.add_argument('--illumination_local_bright_strength_min', type=float, default=0.15,
                        help='局部亮斑乘性增强强度下限。')
    parser.add_argument('--illumination_local_bright_strength_max', type=float, default=0.50,
                        help='局部亮斑乘性增强强度上限。')
    parser.add_argument('--illumination_local_shadow_strength_min', type=float, default=0.10,
                        help='局部阴影乘性衰减强度下限。')
    parser.add_argument('--illumination_local_shadow_strength_max', type=float, default=0.40,
                        help='局部阴影乘性衰减强度上限。')
    parser.add_argument('--illumination_local_shadow_prob', type=float, default=0.5,
                        help='每个局部区域被模拟为阴影而非亮斑的概率。')
    parser.add_argument('--illumination_local_noise_prob', type=float, default=0.5,
                        help='每个局部区域内部叠加低频不均匀光照场的概率。')
    parser.add_argument('--illumination_local_noise_strength', type=float, default=0.25,
                        help='局部区域内部低频不均匀光照场强度。')
    parser.add_argument('--illumination_local_blur_ratio_min', type=float, default=0.35,
                        help='局部亮斑/阴影边缘模糊比例下限。')
    parser.add_argument('--illumination_local_blur_ratio_max', type=float, default=0.85,
                        help='局部亮斑/阴影边缘模糊比例上限。')
    parser.add_argument('--illumination_exposure_min', type=float, default=0.45,
                        help='曝光增强乘性因子下限，小于 1 表示欠曝光。')
    parser.add_argument('--illumination_exposure_max', type=float, default=1.8,
                        help='曝光增强乘性因子上限，大于 1 表示过曝光。')
    parser.add_argument('--illumination_gamma_min', type=float, default=0.6,
                        help='gamma 增强下限，小于 1 倾向变亮。')
    parser.add_argument('--illumination_gamma_max', type=float, default=1.8,
                        help='gamma 增强上限，大于 1 倾向变暗。')
    parser.add_argument('--illumination_contrast_min', type=float, default=0.7,
                        help='对比度增强下限。')
    parser.add_argument('--illumination_contrast_max', type=float, default=1.5,
                        help='对比度增强上限。')
    parser.add_argument('--illumination_channel_gain_min', type=float, default=0.85,
                        help='RGB channel gain 下限，用于模拟轻微色温/通道变化。')
    parser.add_argument('--illumination_channel_gain_max', type=float, default=1.15,
                        help='RGB channel gain 上限，用于模拟轻微色温/通道变化。')
    parser.add_argument('--illumination_local_prob', type=float, default=0.5,
                        help='应用局部低频光照场的概率。')
    parser.add_argument('--illumination_local_strength_min', type=float, default=0.0,
                        help='局部低频光照场强度下限。')
    parser.add_argument('--illumination_local_strength_max', type=float, default=0.35,
                        help='局部低频光照场强度上限。')
    parser.add_argument('--train_num_workers', type=int, default=4,
                        help='训练 DataLoader 的 num_workers。')
    parser.add_argument('--eval_num_workers', type=int, default=4,
                        help='测试/验证/true_val DataLoader 的 num_workers,默认 0 以降低内存和磁盘 IO。')
    parser.add_argument('--pin_memory', action='store_true',
                        help='是否为 DataLoader 开启 pin_memory。内存紧张时不要开启。')

    parser.add_argument('--final_threshold_json_path', type=str, default='./final_thresholds_by_split.json',
                        help='保存最终阈值json，格式如 can -> test_private / test_private_mixed')

    parser.add_argument('--dinov3_repo', type=str, default='./dinov3')
    parser.add_argument('--dinov3_weights', type=str, default='./backbones/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
    parser.add_argument('--dinov3_check_hash', action='store_true')
    parser.add_argument('--dinov3_force_reload', action='store_true')

    parser.add_argument('--item_list', nargs='+', default=['sheet_metal'], help='item列表（空格分隔）')


    args = parser.parse_args()

    args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_INP_num={args.INP_num}_Seg={args.seg_hidden_dim}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # category info
    main(args)