import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
import random
from functools import partial
from typing import Iterator, List, Optional, Sequence, Tuple
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
import gc
from optimizers import StableAdamW
from utils import evaluation_batch, WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger, ader_evaluator

# Dataset-Related Modules
from mvtec2_dataset import MVTec2Dataset
from mvtec2_dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from utils import get_gaussian_kernel, cal_anomaly_maps, setup_seed_strict
from utils2 import DTDTextureBank, dice_bce_loss, synthesize_dtd_perlin

# Model-Related Modules
from models import vit_encoder
from models.uad2 import INP_Former, ResidualSegHead
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from PIL import Image, ImageEnhance
import cv2


warnings.filterwarnings("ignore")

CROP_COORD_PATTERN = re.compile(r"^(?P<base>.+)_x(?P<x1>\d+)_y(?P<y1>\d+)_x(?P<x2>\d+)_y(?P<y2>\d+)$")
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


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



def _is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMG_EXTENSIONS)


def _list_images_recursive(root: str) -> List[str]:
    results = []
    if not os.path.isdir(root):
        return results
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if _is_image_file(fname):
                results.append(os.path.join(dirpath, fname))
    results.sort()
    return results


def _truncate_paths(paths: List[str], max_samples: int, seed: int = 1) -> List[str]:
    if max_samples is None or max_samples <= 0 or len(paths) <= max_samples:
        return paths
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    return paths[:max_samples]


class LocalSynthAnomalyDataset(Dataset):
    def __init__(self, root: str, transform, gt_transform, max_samples: int = 0):
        self.root = os.path.expanduser(root)
        self.images_root = os.path.join(self.root, 'images')
        self.masks_root = os.path.join(self.root, 'masks')
        self.transform = transform
        self.gt_transform = gt_transform

        if not os.path.isdir(self.images_root):
            raise FileNotFoundError(f'images folder not found under: {self.root}')
        if not os.path.isdir(self.masks_root):
            raise FileNotFoundError(f'masks folder not found under: {self.root}')

        self.samples: List[Tuple[str, str]] = []
        for img_path in _list_images_recursive(self.images_root):
            rel = os.path.relpath(img_path, self.images_root)
            mask_path = os.path.join(self.masks_root, rel)
            if not os.path.isfile(mask_path):
                stem = os.path.splitext(rel)[0]
                candidates = [os.path.join(self.masks_root, stem + ext) for ext in IMG_EXTENSIONS]
                mask_path = next((c for c in candidates if os.path.isfile(c)), None)
            if mask_path is None or not os.path.isfile(mask_path):
                raise FileNotFoundError(f'Cannot find matching mask for synthetic anomaly image: {img_path}')
            self.samples.append((img_path, mask_path))

        self.samples = _truncate_paths(sorted(self.samples), max_samples)
        if len(self.samples) == 0:
            raise FileNotFoundError(f'No synthetic anomaly images found under: {self.root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        with Image.open(img_path).convert('RGB') as img:
            img = self.transform(img) if self.transform is not None else img
        with Image.open(mask_path).convert('L') as mask:
            mask = self.gt_transform(mask) if self.gt_transform is not None else mask
        mask = (mask > 0.5).float()
        label = np.int64(1)
        return img, mask, label, img_path


def _cycle_loader(dataloader: DataLoader) -> Iterator:
    while True:
        for batch in dataloader:
            yield batch


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



def update_running_stats(stats, anomaly_map, resize_mask=None):
    anomaly_map = np.clip(np.asarray(anomaly_map, dtype=np.float32), 0, 1)

    if resize_mask is not None:
        if isinstance(resize_mask, int):
            target_h, target_w = resize_mask, resize_mask
        else:
            target_h, target_w = resize_mask
        anomaly_map = cv2.resize(anomaly_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    values = anomaly_map.astype(np.float64).reshape(-1)
    stats["sum"] += float(values.sum())
    stats["sumsq"] += float(np.square(values).sum())
    stats["count"] += int(values.size)


def merge_stats(dst, src):
    dst["sum"] += src["sum"]
    dst["sumsq"] += src["sumsq"]
    dst["count"] += src["count"]


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
    save_root,
    data_root,
    merge_mode="average",
    collect_anomaly_scores=False,
    normalize_amap=False,
    stats_resize_mask=None,
):
    model.eval()

    current_record = None
    current_key = None
    stats = {"sum": 0.0, "sumsq": 0.0, "count": 0}

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    def finalize_save_and_release(record):
        merged_record = finalize_single_merge_record(record, merge_mode=merge_mode)
        anomaly_map = np.clip(merged_record["anomaly_map"], 0, 1)

        save_path = build_merged_save_path(
            save_root=save_root,
            img_path=merged_record["first_img_path"],
            base_name=merged_record["base_name"],
            data_root=data_root,
            suffix=".tiff",
        )

        anomaly_map_gray = np.clip(anomaly_map * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(save_path, anomaly_map_gray)

        if collect_anomaly_scores:
            update_running_stats(stats, anomaly_map, resize_mask=stats_resize_mask)

        del merged_record, anomaly_map, anomaly_map_gray

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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_trainable_state(trainable_modules, save_path):
    state_dict = {
        k: v.detach().cpu()
        for k, v in trainable_modules.state_dict().items()
    }
    torch.save(state_dict, save_path)


def load_trainable_state(trainable_modules, load_path, device):
    state_dict = torch.load(load_path, map_location=device)
    trainable_modules.load_state_dict(state_dict, strict=True)

def main(args):
    # Fixing the Random Seed
    setup_seed(1)
    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(1)

    aux_loader_generator = torch.Generator()
    aux_loader_generator.manual_seed(2)

    def get_category_model_path(category):
        safe_category = str(category).replace(os.sep, "_")
        if os.altsep is not None:
            safe_category = safe_category.replace(os.altsep, "_")
        return os.path.join(args.save_dir, args.save_name, f"model_{safe_category}.pth")

    if is_dinov3_encoder(args.encoder):
        args.input_size = make_divisible_down(args.input_size, 16)
        args.crop_size = make_divisible_down(args.crop_size, 16)

    # Data Preparation
    data_transform, gt_transform, data_transforms_train = get_data_transforms(args.input_size, args.crop_size)

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(args.item_list):
        train_path = os.path.join(args.data_path, item)
        test_path = os.path.join(args.data_path, item)

        train_data = MVTec2Dataset(root=train_path, transform=data_transforms_train, phase="train", resize=args.input_size, normal_only=True)
        train_data_list.append(train_data)

        test_data = MVTec2Dataset(root=test_path, transform=data_transform, phase="test", resize=args.input_size, normal_only=True)
        test_data_list.append(test_data)

    aux_normal_count = 0
    aux_anomaly_count = 0
    synth_anomaly_count = 0
    aux_seg_loader = None
    aux_seg_iterator = None
    texture_bank = None

    if args.use_synth_anomalies and args.synth_anomaly_root:
        synth_anomaly_dataset = LocalSynthAnomalyDataset(
            root=args.synth_anomaly_root,
            transform=data_transforms_train,
            gt_transform=gt_transform,
            max_samples=args.synth_anomaly_max_samples,
        )
        synth_anomaly_count = len(synth_anomaly_dataset)
        aux_seg_loader = DataLoader(
            synth_anomaly_dataset,
            batch_size=args.synth_batch_size if args.synth_batch_size > 0 else args.batch_size,
            shuffle=True,
            num_workers=args.train_num_workers, #args.train_num_workers
            drop_last=True,
            pin_memory=args.pin_memory,
            collate_fn=four_tuple_collate,
            worker_init_fn=seed_worker,
            generator=aux_loader_generator,

        )
        aux_seg_iterator = _cycle_loader(aux_seg_loader)
        print_fn(f'synth anomaly images loaded: {synth_anomaly_count}')


    if len(train_data_list) == 0:
        raise RuntimeError('No training data found.')

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=four_tuple_collate,
        worker_init_fn=seed_worker,
        generator=train_loader_generator,
    )

    # Adopting a grouping-based reconstruction strategy similar to Dinomaly

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

    if args.phase == 'train':
        # Model Initialization
        normal_trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
        seg_trainable = nn.ModuleList([residual_head])
        init_modules = nn.ModuleList([normal_trainable, seg_trainable])
        for m in init_modules.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

        optimizer_normal = StableAdamW(
            [{'params': normal_trainable.parameters()}],
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            amsgrad=True,
            eps=1e-10,
        )
        optimizer_seg = StableAdamW(
            [{'params': seg_trainable.parameters()}],
            lr=args.seg_lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            amsgrad=True,
            eps=1e-10,
        )
        total_iters = args.total_epochs * len(train_dataloader)
        lr_scheduler_normal = WarmCosineScheduler(
            optimizer_normal,
            base_value=args.lr,
            final_value=args.final_lr,
            total_iters=total_iters,
            warmup_iters=100,
        )
        lr_scheduler_seg = WarmCosineScheduler(
            optimizer_seg,
            base_value=args.seg_lr,
            final_value=args.seg_final_lr,
            total_iters=max(1, total_iters // max(1, args.seg_update_interval)),
            warmup_iters=20,
        )
        print_fn(
            'train image number:{} (target_normal + aux_normal={}, synth_anomaly={}, aux_anomaly={})'.format(
                len(train_data), aux_normal_count, synth_anomaly_count, aux_anomaly_count
            )
        )

        # Train
        global_step = 0
        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            recon_loss_list = []
            seg_loss_list = []

            for img, gt, label, _ in tqdm(train_dataloader, ncols=80):
                img = img.to(device)

                optimizer_normal.zero_grad(set_to_none=True)
                en, de, g_loss, _ = model(img, return_seg=False)
                recon_loss = global_cosine_hm_adaptive(en, de, y=3)
                normal_loss = args.recon_loss_weight * recon_loss + args.gather_loss_weight * g_loss
                normal_loss.backward()
                nn.utils.clip_grad_norm_(normal_trainable.parameters(), max_norm=0.1)
                optimizer_normal.step()
                lr_scheduler_normal.step()

                seg_loss = torch.zeros(1, device=device)
                do_seg_step = (
                    (epoch + 1) >= args.residual_start_epoch
                    and ((global_step + 1) % max(1, args.seg_update_interval) == 0)
                )
                if do_seg_step:
                    optimizer_seg.zero_grad(set_to_none=True)
                    if aux_seg_iterator is not None:
                        seg_img, seg_mask, _, _ = next(aux_seg_iterator)
                        seg_img = seg_img.to(device)
                        seg_mask = seg_mask.to(device)
                    elif texture_bank is not None:
                        seg_img, seg_mask = synthesize_dtd_perlin(
                            img,
                            texture_bank,
                            beta_min=args.dtd_beta_min,
                            beta_max=args.dtd_beta_max,
                            no_anomaly_prob=args.dtd_no_anomaly_prob,
                            scale_min=args.perlin_scale_min,
                            scale_max=args.perlin_scale_max,
                        )
                    else:
                        seg_img = None
                        seg_mask = None

                    if seg_img is not None:
                        seg_logits = model.forward_seg(
                            seg_img,
                            seg_out_size=seg_img.shape[-2:],
                            freeze_backbone=True,
                        )
                        seg_loss = dice_bce_loss(seg_logits, seg_mask)
                        (args.seg_loss_weight * seg_loss).backward()
                        nn.utils.clip_grad_norm_(seg_trainable.parameters(), max_norm=0.1)
                        optimizer_seg.step()
                        lr_scheduler_seg.step()

                total_loss = normal_loss.detach() + args.seg_loss_weight * seg_loss.detach()
                loss_list.append(total_loss.item())
                recon_loss_list.append(recon_loss.item())
                seg_loss_list.append(seg_loss.item())
                global_step += 1

            print_fn(
                'epoch [{}/{}], loss:{:.4f}, recon:{:.4f}, seg:{:.4f}'.format(
                    epoch + 1,
                    args.total_epochs,
                    np.mean(loss_list),
                    np.mean(recon_loss_list),
                    np.mean(seg_loss_list),
                )
            )

            if (epoch + 1) % args.total_epochs == 0:
                category = args.item_list[0]
                model_path = get_category_model_path(category)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                save_trainable_state(trainable_modules, model_path)
                print_fn(f"Saved trainable modules to {model_path}")
                model.train()
    elif args.phase == 'test':
        save_dir = './results/anomaly_images'
        os.makedirs(save_dir, exist_ok=True)  # 创建根目录

        for item, val_data in zip(args.item_list, test_data_list):
            model_path = get_category_model_path(item)
            load_trainable_state(trainable_modules, model_path, device)
            model.eval()
            print_fn(f"Loaded trainable modules from {model_path}")
            val_dataloader = make_eval_dataloader(val_data, args)
            save_merged_anomaly_maps(
                model=model,
                dataloader=val_dataloader,
                device=device,
                save_root=save_dir,
                data_root=args.data_path,
                merge_mode=args.merge_mode,
                collect_anomaly_scores=False,
                normalize_amap=args.normalize_amap,
                stats_resize_mask=args.eval_resize_mask,
            )

            del val_dataloader
            gc.collect()


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'Mvtec_ad_2') # 'MVTec-AD' or 'VisA' or 'Real-IAD'
    parser.add_argument('--data_path', type=str, default=r'./datasets/mvtec_ad_2_splits_1024') # Replace it with your path.

    parser.add_argument('--use_synth_anomalies', action='store_true', help='Use local synthetic anomaly images/masks to train the segmentation head')
    parser.add_argument('--synth_anomaly_root', type=str, default='./datasets/synthesized_mvtecad2_1024rgbl/fruit_jelly', help='Expected structure: root/images and root/masks with matching filenames')
    parser.add_argument('--synth_anomaly_max_samples', type=int, default=512)
    parser.add_argument('--synth_batch_size', type=int, default=8)

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='INP-Former-Multi-Class')

    # model info
    parser.add_argument('--encoder', type=str, default='dinov3_vith16plus') 
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--seg_hidden_dim', type=int, default=512)

    # training info
    parser.add_argument('--total_epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--phase', type=str, default='train') # true_val
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--final_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--recon_loss_weight', type=float, default=1.0)
    parser.add_argument('--gather_loss_weight', type=float, default=0.2)
    parser.add_argument('--seg_loss_weight', type=float, default=0.5)
    parser.add_argument('--eval_seg_weight', type=float, default=0.03)
    parser.add_argument('--residual_start_epoch', type=int, default=1)
    parser.add_argument('--seg_update_interval', type=int, default=2)
    parser.add_argument('--seg_lr', type=float, default=3e-4)
    parser.add_argument('--seg_final_lr', type=float, default=1e-5)

    parser.add_argument('--merge_mode', type=str, default='average', choices=['average', 'max', 'overwrite'],
                        help='merge crop anomaly maps before metric/threshold/mean/std calculation')
    parser.add_argument('--eval_resize_mask', type=int, default=256,
                        help='先 merge 到原图空间，再 resize 到该尺寸后调用 ader_evaluator；设为小于等于0表示不 resize。')
    parser.add_argument('--train_num_workers', type=int, default=4,
                        help='训练 DataLoader 的 num_workers。')
    parser.add_argument('--eval_num_workers', type=int, default=4,
                        help='测试/验证/true_val DataLoader 的 num_workers,默认 0 以降低内存和磁盘 IO。')
    parser.add_argument('--pin_memory', action='store_true',
                        help='是否为 DataLoader 开启 pin_memory。内存紧张时不要开启。')
    parser.add_argument('--normalize_amap', action='store_true',)

    parser.add_argument('--dinov3_repo', type=str, default='./dinov3')
    parser.add_argument('--dinov3_weights', type=str, default='./backbones/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
    parser.add_argument('--dinov3_check_hash', action='store_true')
    parser.add_argument('--dinov3_force_reload', action='store_true')

    parser.add_argument('--item_list', nargs='+', default=['fruit_jelly'], help='item列表（空格分隔）')
    # parser.add_argument('--item_list', nargs='+', default=['can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts'], help='item列表（空格分隔）')


    args = parser.parse_args()

    if args.eval_resize_mask <= 0:
        args.eval_resize_mask = None

    args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_INP_num={args.INP_num}_Seg={args.seg_hidden_dim}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # category info
    main(args)