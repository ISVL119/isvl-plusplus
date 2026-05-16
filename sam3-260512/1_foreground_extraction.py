import os

from glob import glob
from itertools import chain
from pprint import pformat
import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.append('./')

from loguru import logger
from tqdm import tqdm
import cv2 as cv
import numpy as np
import torch
from PIL import Image, ImageOps

from dataset import DATASET_INFOS, test_transform, read_image
from models import MODEL_INFOS, BaseModel
from models.feb import get_feb
from utils import save_dependencies_files, fix_seeds


try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    raise ImportError(
        f"导入 SAM3 模块失败: {e}\n"
        f"请确保已正确安装 SAM3 依赖，并且 sam3 模块在当前项目或 PYTHONPATH 中。"
    )


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resize_foreground_to_original(foreground: np.ndarray, image_path: str):
    original_image = read_image(image_path)
    original_h, original_w = original_image.shape[:2]

    if foreground.shape[:2] != (original_h, original_w):
        foreground = cv.resize(
            foreground,
            (original_w, original_h),
            interpolation=cv.INTER_LINEAR
        )

    return foreground


def load_image_rgb(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def state_to_mask_list(state):
    for key in ("masks", "out_binary_masks"):
        if key not in state:
            continue

        masks = state[key]
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        elif isinstance(masks, list):
            masks = [
                m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else m
                for m in masks
            ]
            masks = np.array(masks, dtype=object)
        elif isinstance(masks, tuple):
            masks = np.array(list(masks), dtype=object)

        if isinstance(masks, np.ndarray) and masks.dtype != object:
            if masks.ndim == 4:
                return [np.asarray(m.squeeze()) for m in masks]
            if masks.ndim == 3:
                return [np.asarray(m) for m in masks]
            if masks.ndim == 2:
                return [np.asarray(masks)]

        if isinstance(masks, np.ndarray) and masks.dtype == object:
            out = []
            for m in masks.tolist():
                arr = np.asarray(m)
                out.append(arr.squeeze())
            return out

    return []


def build_combined_mask(mask_list, image_hw, mask_threshold):
    height, width = image_hw
    combined = np.zeros((height, width), dtype=bool)

    for mask in mask_list:
        mask_arr = np.asarray(mask).squeeze()
        if mask_arr.ndim != 2:
            continue

        if mask_arr.shape != (height, width):
            mask_arr = cv.resize(
                mask_arr.astype(np.float32),
                (width, height),
                interpolation=cv.INTER_LINEAR,
            )

        if mask_arr.dtype == np.bool_:
            binary = mask_arr
        else:
            binary = mask_arr > mask_threshold

        combined |= binary

    return combined.astype(np.uint8) * 255


def initialize_sam3_image_predictor(checkpoint_path, bpe_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM3 模型文件不存在: {checkpoint_path}")
    if not os.path.exists(bpe_path):
        raise FileNotFoundError(f"SAM3 BPE 文件不存在: {bpe_path}")

    image_model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        device=DEVICE,
    )
    return Sam3Processor(image_model, device=DEVICE)


def predict_sam3_mask(
    image_predictor,
    image_path,
    text_prompt,
    confidence_threshold,
    mask_threshold,
):
    image = load_image_rgb(image_path)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if DEVICE == "cuda"
        else nullcontext()
    )

    with autocast_context:
        state = image_predictor.set_image(image)
        state = image_predictor.set_text_prompt(text_prompt, state)
        state = image_predictor.set_confidence_threshold(confidence_threshold, state)

    mask_list = state_to_mask_list(state)
    combined_mask = build_combined_mask(
        mask_list,
        image_hw=(image.height, image.width),
        mask_threshold=mask_threshold,
    )

    return combined_mask, len(mask_list)


def collect_predict_files(root_dir):
    predict_fns = (
        sorted(glob(os.path.join(root_dir, 'train/*/*')))
    )

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    predict_fns = [
        fn for fn in predict_fns
        if os.path.isfile(fn) and os.path.splitext(fn)[1].lower() in image_exts
    ]

    return sorted(set(predict_fns))


def remove_small_components(mask: np.ndarray, min_area: int):
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=4)
    cleaned = np.zeros_like(mask)
    foreground = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 255
            foreground += area

    return cleaned, foreground


def binarize_probability_map(
    foreground: np.ndarray,
    close_radius: int = 3,
    open_radius: int = 1,
    min_area_ratio: float = 0.0005,
):
    foreground = np.squeeze(foreground)
    foreground = np.nan_to_num(foreground)

    if foreground.ndim != 2:
        raise ValueError(f"Expected 2D foreground map, got shape={foreground.shape}")

    foreground = foreground.astype(np.float32)

    if foreground.max() <= 1.0 and foreground.min() >= 0.0:
        gray = (np.clip(foreground, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        gray = np.clip(foreground, 0.0, 255.0).astype(np.uint8)

    _, mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    if close_radius > 0:
        k = close_radius * 2 + 1
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (k, k))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    if open_radius > 0:
        k = open_radius * 2 + 1
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (k, k))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    min_area = max(64, round(mask.shape[0] * mask.shape[1] * min_area_ratio))
    mask, _ = remove_small_components(mask, min_area)

    return mask


def fill_holes(mask: np.ndarray, max_hole_area_ratio: float = 0.1):
    mask = np.squeeze(mask)
    mask = np.nan_to_num(mask)

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")

    binary = (mask > 0).astype(np.uint8)

    h, w = binary.shape
    image_area = h * w
    max_hole_area = int(image_area * max_hole_area_ratio)

    background = (binary == 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        background,
        connectivity=4
    )

    filled = binary.copy()

    for label in range(1, num_labels):
        x = int(stats[label, cv.CC_STAT_LEFT])
        y = int(stats[label, cv.CC_STAT_TOP])
        bw = int(stats[label, cv.CC_STAT_WIDTH])
        bh = int(stats[label, cv.CC_STAT_HEIGHT])
        area = int(stats[label, cv.CC_STAT_AREA])

        touches_border = (
            x == 0 or
            y == 0 or
            x + bw >= w or
            y + bh >= h
        )

        if touches_border:
            continue

        if area > max_hole_area:
            continue

        filled[labels == label] = 1

    return filled.astype(np.float32)


def save_foreground_png_and_npy(foreground, save_dir, image_name, save_npy=True):
    os.makedirs(save_dir, exist_ok=True)

    foreground = np.asarray(foreground)
    if foreground.dtype == np.bool_:
        foreground_png = foreground.astype(np.uint8) * 255
        foreground_npy = foreground.astype(np.float32)
    elif foreground.dtype == np.uint8:
        foreground_png = foreground
        foreground_npy = (foreground.astype(np.float32) / 255.0)
    else:
        foreground = np.clip(foreground.astype(np.float32), 0.0, 1.0)
        foreground_png = (foreground * 255.0).astype(np.uint8)
        foreground_npy = foreground

    cv.imwrite(os.path.join(save_dir, f'f_{image_name}.png'), foreground_png)

    if save_npy:
        np.save(os.path.join(save_dir, f'f_{image_name}.npy'), foreground_npy)


@torch.no_grad()
def build_fallback_foreground_estimator(
    root_dir,
    model_name,
    layer,
    resize,
    vis,
    device,
):
    logger.info('build fallback foreground estimator')

    model: BaseModel = MODEL_INFOS[model_name]['cls']([layer], input_size=resize).to(device)
    model.eval()

    train_image = {}
    train_ks = []
    train_image_fns = sorted(glob(os.path.join(root_dir, 'train/*/*')))
    train_features = torch.zeros(len(train_image_fns), *model.shapes[0][1:], device=device)

    for i, fn in enumerate(tqdm(train_image_fns, desc='extract train features', leave=False)):
        assert os.path.exists(fn), f'{fn} not exists'
        k = os.path.relpath(fn, root_dir)
        train_ks.append(k)

        image = read_image(fn, (resize, resize))
        image_t = test_transform(image)
        feature = model(image_t[None].to(device))[0]
        train_features[i:i + 1] = feature.detach()

        if vis:
            train_image[k] = image

    feb = get_feb(train_features).to(device).eval()
    return model, feb


@torch.no_grad()
def gen_foreground(
    save_path,
    dataset_path,
    dataset_name,
    dataset_root,
    model_name,
    layer,
    resize,
    vis,
    sam3_checkpoint,
    sam3_bpe_path,
    sam3_threshold,
    sam3_mask_threshold,
    save_npy,
):
    device = torch.device('cuda')
    logger.info(f'gen_foreground')
    logger.info(f'save to {save_path}')
    logger.info(f'params: {dataset_name} {model_name} {layer} {resize} {vis}')
    logger.info(f'sam3: checkpoint={sam3_checkpoint}, bpe={sam3_bpe_path}, threshold={sam3_threshold}, mask_threshold={sam3_mask_threshold}')

    dataset_info = DATASET_INFOS[dataset_root]
    sam3_predictor = initialize_sam3_image_predictor(
        checkpoint_path=sam3_checkpoint,
        bpe_path=sam3_bpe_path,
    )

    for sub_category in dataset_info[1]:
        fix_seeds(66)

        root_dir = os.path.join(dataset_path, dataset_name, sub_category)
        logger.info(f'generate {sub_category}')
        cur_target_save_path = os.path.join(save_path, sub_category)
        os.makedirs(cur_target_save_path, exist_ok=True)

        logger.info('predict foreground by SAM3 first, fallback to FEB if SAM3 mask is empty')
        predict_fns = collect_predict_files(root_dir)

        model = None
        feb = None
        sam3_success_count = 0
        fallback_count = 0

        for fn in tqdm(predict_fns, desc='predict data', leave=False):
            assert os.path.exists(fn), f'{fn} not exists'
            k = os.path.relpath(fn, root_dir)

            cur_save_dir = os.path.dirname(os.path.join(cur_target_save_path, k))
            cur_image_name = os.path.basename(k).split('.', 1)[0]

            sam3_mask, candidate_count = predict_sam3_mask(
                image_predictor=sam3_predictor,
                image_path=fn,
                text_prompt=sub_category,
                confidence_threshold=sam3_threshold,
                mask_threshold=sam3_mask_threshold,
            )

            if bool((sam3_mask > 0).any()):
                sam3_mask = fill_holes(sam3_mask)

                save_foreground_png_and_npy(
                    foreground=sam3_mask,
                    save_dir=cur_save_dir,
                    image_name=cur_image_name,
                    save_npy=save_npy,
                )
                sam3_success_count += 1
                continue

            if feb is None:
                model, feb = build_fallback_foreground_estimator(
                    root_dir=root_dir,
                    model_name=model_name,
                    layer=layer,
                    resize=resize,
                    vis=vis,
                    device=device,
                )

            image = read_image(fn, (resize, resize))
            image_t = test_transform(image)
            feature = model(image_t[None].to(device))[0]

            foreground = feb(feature)[0, 0].cpu().numpy()
            foreground = resize_foreground_to_original(foreground, fn)
            foreground = binarize_probability_map(foreground)
            foreground = fill_holes(foreground)

            save_foreground_png_and_npy(
                foreground=foreground,
                save_dir=cur_save_dir,
                image_name=cur_image_name,
                save_npy=save_npy,
            )
            fallback_count += 1

        logger.info(
            f'{sub_category}: total={len(predict_fns)}, '
            f'sam3_success={sam3_success_count}, fallback={fallback_count}'
        )

        if feb is not None:
            torch.save(feb, os.path.join(cur_target_save_path, f'feb.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # run
    parser.add_argument("-op", "--out-path", type=str, default="./log/foreground/foreground_mvtecad2_keepsize_sam3", help="log path")
    # data
    parser.add_argument("--dataset-path", type=str, default="../datasets/")
    parser.add_argument("--dataset-name", type=str, default="mvtec_ad_2")
    parser.add_argument("--dataset-root", type=str, default="mvtec_ad_2")
    parser.add_argument("--resize", type=int, default=512, help="image resize")
    # vis
    parser.add_argument("--vis", action="store_true", help='kept for compatibility; png foreground is always saved')
    parser.add_argument("--save-npy", action="store_true", help="also save foreground as npy for downstream code compatibility")
    # model
    parser.add_argument("-pm", "--pretrained-model", type=str, default='DenseNet', choices=list(MODEL_INFOS.keys()), help="pretrained model")
    parser.add_argument("--layer", type=str, default='features.denseblock1', choices=list(chain(*[v['layers'] for k, v in MODEL_INFOS.items()])), help=f'feature layer, ' + ", ".join([f"{k}: {v['layers']}" for k, v in MODEL_INFOS.items()]))

    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_sam3_checkpoint = os.path.join(current_dir, "..", "backbones", "weights","sam3.pt")
    default_sam3_bpe = os.path.join(current_dir, "assets", "bpe_simple_vocab_16e6.txt.gz")

    parser.add_argument("--sam3-checkpoint", type=str, default=default_sam3_checkpoint, help="SAM3 model checkpoint path")
    parser.add_argument("--sam3-bpe-path", type=str, default=default_sam3_bpe, help="SAM3 BPE vocab path")
    parser.add_argument("--sam3-threshold", type=float, default=0.30, help="SAM3 confidence threshold")
    parser.add_argument("--sam3-mask-threshold", type=float, default=0.0, help="SAM3 mask binarization threshold")

    args = parser.parse_args()
    # check
    if args.layer not in MODEL_INFOS[args.pretrained_model]['layers']:
        parser.error(f'{args.layer} not in {MODEL_INFOS[args.pretrained_model]["layers"]}')
    if args.out_path is None:
        args.out_path = f'log/foreground/foreground_{args.dataset_name}'
    if not 0.0 <= args.sam3_threshold <= 1.0:
        parser.error('--sam3-threshold must be in [0, 1]')

    # script_dir = Path(__file__).parent.resolve()
    # out_path = (script_dir / args.out_path).resolve()
    os.makedirs(args.out_path, exist_ok=True)
    logger.add(os.path.join(args.out_path, 'runtime.log'))

    logger.info('args: \n' + pformat(vars(args)))
    assert torch.cuda.is_available(), f'cuda is not available'
    save_dependencies_files(os.path.join(args.out_path, 'src'))
    gen_foreground(
        args.out_path,
        args.dataset_path,
        args.dataset_name,
        args.dataset_root,
        args.pretrained_model,
        args.layer,
        args.resize,
        args.vis,
        args.sam3_checkpoint,
        args.sam3_bpe_path,
        args.sam3_threshold,
        args.sam3_mask_threshold,
        args.save_npy,
    )