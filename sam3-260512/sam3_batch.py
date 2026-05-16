#!/usr/bin/env python3
"""
SAM3 批量文本分割脚本

功能：
1. 对指定文件夹中的图片进行批量处理
2. 使用文本 prompt 进行 SAM3 分割
3. 将分割得到的二值 mask 保存到指定输出文件夹
4. 通过命令行参数调节阈值

示例：
python batch_sam3.py \
    --input-dir /path/to/images \
    --output-dir /path/to/masks \
    --prompt "a cat" \
    --threshold 0.40 \
    --mask-threshold 0.0
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"导入 SAM3 模块失败: {e}", file=sys.stderr)
    print("请确保已正确安装 SAM3 依赖。", file=sys.stderr)
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def list_image_files(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"输入文件夹不存在: {folder_path}")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"输入路径不是文件夹: {folder_path}")

    image_files = []
    for name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, name)
        if os.path.isfile(full_path) and is_image_file(full_path):
            image_files.append(full_path)
    return sorted(image_files)


def load_image_rgb(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image, np.array(image)


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
    """
    将所有候选 mask 按位求并，输出单张二值 mask。
    """
    height, width = image_hw
    combined = np.zeros((height, width), dtype=bool)

    for mask in mask_list:
        mask_arr = np.asarray(mask).squeeze()
        if mask_arr.ndim != 2:
            continue

        if mask_arr.shape != (height, width):
            mask_arr = cv2.resize(
                mask_arr.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        if mask_arr.dtype == np.bool_:
            binary = mask_arr
        else:
            binary = mask_arr > mask_threshold

        combined |= binary

    return combined.astype(np.uint8) * 255


def initialize_image_predictor(checkpoint_path, bpe_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    if not os.path.exists(bpe_path):
        raise FileNotFoundError(f"BPE 文件不存在: {bpe_path}")

    image_model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        device=DEVICE,
    )
    return Sam3Processor(image_model, device=DEVICE)


def predict_mask(image_predictor, image, text_prompt, confidence_threshold, mask_threshold):
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


def save_mask(mask, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)


def process_folder(
    input_dir,
    output_dir,
    text_prompt,
    confidence_threshold,
    mask_threshold,
    checkpoint_path,
    bpe_path,
    save_empty_mask,
):
    image_files = list_image_files(input_dir)
    if not image_files:
        raise RuntimeError(f"输入文件夹中没有可处理的图片: {input_dir}")

    image_predictor = initialize_image_predictor(checkpoint_path, bpe_path)

    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    failed_items = []
    empty_count = 0
    start_time = time.time()

    print(f"使用设备: {DEVICE}")
    print(f"输入目录: {os.path.abspath(input_dir)}")
    print(f"输出目录: {os.path.abspath(output_dir)}")
    print(f"图片数量: {len(image_files)}")
    print(f"文本提示: {text_prompt}")
    print(f"置信度阈值: {confidence_threshold:.4f}")
    print(f"Mask 二值化阈值: {mask_threshold:.4f}")
    print("-" * 80)

    for idx, image_path in enumerate(image_files, start=1):
        try:
            image, _ = load_image_rgb(image_path)
            combined_mask, candidate_count = predict_mask(
                image_predictor=image_predictor,
                image=image,
                text_prompt=text_prompt,
                confidence_threshold=confidence_threshold,
                mask_threshold=mask_threshold,
            )

            has_foreground = bool((combined_mask > 0).any())
            if not has_foreground:
                empty_count += 1
                if not save_empty_mask:
                    print(f"[{idx}/{len(image_files)}] 跳过空 mask: {os.path.basename(image_path)}")
                    continue

            stem, _ = os.path.splitext(os.path.basename(image_path))
            save_path = os.path.join(output_dir, f"{stem}_mask.png")
            save_mask(combined_mask, save_path)
            success_count += 1

            print(
                f"[{idx}/{len(image_files)}] 完成: {os.path.basename(image_path)} -> {os.path.basename(save_path)} | "
                f"候选 mask 数: {candidate_count} | 前景像素: {int((combined_mask > 0).sum())}"
            )

        except Exception as e:
            failed_items.append(f"{os.path.basename(image_path)}: {e}")
            print(
                f"[{idx}/{len(image_files)}] 失败: {os.path.basename(image_path)} | {e}",
                file=sys.stderr,
            )

    elapsed = time.time() - start_time
    print("-" * 80)
    print("处理完成")
    print(f"总图片数: {len(image_files)}")
    print(f"成功保存: {success_count}")
    print(f"空 mask 数: {empty_count}")
    print(f"失败数量: {len(failed_items)}")
    print(f"总耗时: {elapsed:.2f}s")

    if failed_items:
        print("\n失败明细:", file=sys.stderr)
        for item in failed_items:
            print(f"- {item}", file=sys.stderr)


def build_argparser():
    default_checkpoint = os.path.join(current_dir, "models", "sam3.pt") # 权重文件目录
    default_bpe = os.path.join(current_dir, "assets", "bpe_simple_vocab_16e6.txt.gz") # bpe文件

    parser = argparse.ArgumentParser(
        description="基于 SAM3 的批量文本分割脚本：对文件夹中的图片进行分割并保存 mask。"
    )
    parser.add_argument(
        "--input-dir",
        default="E:/dataset/mvtec_ad_2/wallplugs/train/good", #改成你的目录
        help="输入图片文件夹路径",
    )
    parser.add_argument(
        "--output-dir",
        default="./output_masks",
        help="输出 mask 文件夹路径",
    )
    parser.add_argument(
        "--prompt",
        default="wallplugs", # central dark textured rectangular region
        help="文本 prompt，例如: 'a cat'",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="SAM3 置信度阈值，默认 0.40",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.0,
        help="mask 二值化阈值，默认 0.0（与原始代码中的 mask > 0 保持一致）",
    )
    parser.add_argument(
        "--checkpoint",
        default=default_checkpoint,
        help=f"SAM3 模型权重路径，默认: {default_checkpoint}",
    )
    parser.add_argument(
        "--bpe-path",
        default=default_bpe,
        help=f"BPE 文件路径，默认: {default_bpe}",
    )
    parser.add_argument(
        "--no-save-empty-mask",
        action="store_true",
        help="如果图片没有分割到前景，则不保存全黑 mask",
    )
    return parser


def validate_args(args):
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold 必须在 [0, 1] 范围内")
    if not args.prompt or not args.prompt.strip():
        raise ValueError("--prompt 不能为空")
    if not args.input_dir or not args.input_dir.strip():
        raise ValueError("--input-dir 不能为空")
    if not args.output_dir or not args.output_dir.strip():
        raise ValueError("--output-dir 不能为空")


def main():
    parser = build_argparser()
    args = parser.parse_args()

    try:
        validate_args(args)
        process_folder(
            input_dir=args.input_dir.strip(),
            output_dir=args.output_dir.strip(),
            text_prompt=args.prompt.strip(),
            confidence_threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            checkpoint_path=args.checkpoint,
            bpe_path=args.bpe_path,
            save_empty_mask=not args.no_save_empty_mask,
        )
    except Exception as e:
        print(f"运行失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()