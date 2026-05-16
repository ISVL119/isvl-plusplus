# sam3_batch_mask.py
from __future__ import annotations

import argparse
import inspect
import os
import sys
from typing import Optional, Sequence

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用本地 SAM3 .pt checkpoint，对图片文件夹进行文本前景分割并输出二值 mask。"
    )
    parser.add_argument("--input_dir", type=str, default=r"E:/dataset/mvtec_ad_2/sheet_metal/train/good", help="输入图片文件夹")
    parser.add_argument("--output_dir", type=str, default=r"E:/dataset/mvtec_ad_2/output", help="输出 mask 文件夹")
    parser.add_argument("--checkpoint_path", type=str, default="models/sam3.pt", help="本地 SAM3 .pt 权重路径")
    parser.add_argument("--prompt", type=str, default="gray sheet_metal", help="文本 prompt，例如 'person'")
    parser.add_argument(
        "--bpe_path",
        type=str,
        default=None,
        help="可选：本地 BPE 文件路径（bpe_simple_vocab_16e6.txt.gz）；不传则自动尝试查找",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备，例如 cuda、cuda:0、cpu；默认自动选择",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="实例保留分数阈值",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="mask 二值化阈值",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".png",
        help="输出 mask 文件名后缀",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="若输出文件已存在则跳过",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="遇到单张图片报错时立即停止",
    )
    return parser.parse_args()


def choose_device(device: Optional[str]) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def list_images(input_dir: str) -> list[str]:
    images = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTS:
                images.append(os.path.join(root, name))
    images.sort()
    return images


def to_numpy(x):
    if x is None:
        return None
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass

    if isinstance(x, np.ndarray):
        return x

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.empty((0,), dtype=np.float32)
        return np.array([to_numpy(v) for v in x])

    return np.asarray(x)


def resolve_bpe_path(user_bpe_path: Optional[str]) -> Optional[str]:
    if user_bpe_path is not None:
        if os.path.exists(user_bpe_path):
            return os.path.abspath(user_bpe_path)
        raise FileNotFoundError(f"BPE 文件不存在: {user_bpe_path}")

    candidates: list[str] = []

    try:
        import sam3

        sam3_root = os.path.dirname(os.path.abspath(sam3.__file__))
        candidates.extend(
            [
                os.path.join(sam3_root, "assets", BPE_FILENAME),
                os.path.join(os.path.dirname(sam3_root), "assets", BPE_FILENAME),
                os.path.join(os.path.dirname(os.path.dirname(sam3_root)), "assets", BPE_FILENAME),
            ]
        )
    except Exception:
        pass

    candidates.extend(
        [
            os.path.join(os.getcwd(), "assets", BPE_FILENAME),
            os.path.join(os.getcwd(), "sam3", "assets", BPE_FILENAME),
            os.path.join(sys.prefix, "assets", BPE_FILENAME),
        ]
    )

    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)

    return None


def normalize_mask_array(masks_np: np.ndarray) -> np.ndarray:
    m = np.asarray(masks_np)

    if m.size == 0:
        return m

    if m.ndim == 4:
        if m.shape[1] == 1:
            m = m[:, 0]
        elif m.shape[0] == 1:
            m = m[0]

    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]

    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]

    return m


def merge_masks_to_union(
    masks,
    image_hw: Sequence[int],
    scores=None,
    score_threshold: Optional[float] = None,
    mask_threshold: float = 0.5,
) -> np.ndarray:
    h, w = int(image_hw[0]), int(image_hw[1])
    empty = np.zeros((h, w), dtype=np.uint8)

    if masks is None:
        return empty

    m = to_numpy(masks)
    if m is None or m.size == 0:
        return empty

    m = normalize_mask_array(m)

    s = None
    if scores is not None:
        s = to_numpy(scores)
        if s is not None:
            s = np.asarray(s).reshape(-1)

    if m.ndim == 2:
        if s is not None and score_threshold is not None and s.size >= 1 and float(s[0]) < score_threshold:
            return empty
        if m.dtype == np.bool_:
            return m.astype(np.uint8)
        return (m > mask_threshold).astype(np.uint8)

    if m.ndim == 3:
        if s is not None and score_threshold is not None and s.size == m.shape[0]:
            keep = s >= score_threshold
            m = m[keep]

        if m.size == 0 or m.shape[0] == 0:
            return empty

        if m.dtype == np.bool_:
            union = np.any(m, axis=0)
        else:
            union = np.any(m > mask_threshold, axis=0)

        if union.shape != (h, w):
            raise ValueError(f"mask 尺寸异常，期望 {(h, w)}，实际 {union.shape}")

        return union.astype(np.uint8)

    raise ValueError(f"无法识别的 mask 维度: {m.shape}")


class LocalSam3Segmenter:
    def __init__(
        self,
        checkpoint_path: str,
        bpe_path: Optional[str],
        device: str,
        score_threshold: float,
        mask_threshold: float,
    ) -> None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        build_sig = inspect.signature(build_sam3_image_model)
        build_kwargs = {}

        if "checkpoint_path" in build_sig.parameters:
            build_kwargs["checkpoint_path"] = checkpoint_path
        if "load_from_HF" in build_sig.parameters:
            build_kwargs["load_from_HF"] = False
        if "device" in build_sig.parameters:
            build_kwargs["device"] = device
        if "eval_mode" in build_sig.parameters:
            build_kwargs["eval_mode"] = True
        if "bpe_path" in build_sig.parameters and bpe_path is not None:
            build_kwargs["bpe_path"] = bpe_path

        self.model = build_sam3_image_model(**build_kwargs)
        if hasattr(self.model, "eval"):
            self.model.eval()

        proc_sig = inspect.signature(Sam3Processor)
        proc_kwargs = {}
        if "confidence_threshold" in proc_sig.parameters:
            proc_kwargs["confidence_threshold"] = score_threshold

        self.processor = Sam3Processor(self.model, **proc_kwargs)
        if hasattr(self.processor, "set_confidence_threshold"):
            self.processor.set_confidence_threshold(score_threshold)

        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold

    def predict_union_mask(self, image: Image.Image, prompt: str) -> np.ndarray:
        state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(state=state, prompt=prompt)

        if isinstance(output, dict):
            masks = output.get("masks")
            scores = output.get("scores")
        else:
            masks = getattr(output, "masks", None)
            scores = getattr(output, "scores", None)

        return merge_masks_to_union(
            masks=masks,
            image_hw=(image.height, image.width),
            scores=scores,
            score_threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
        )


def output_path_for_image(image_path: str, input_dir: str, output_dir: str, suffix: str) -> str:
    rel = os.path.relpath(image_path, input_dir)
    rel_dir = os.path.dirname(rel)
    stem, _ = os.path.splitext(os.path.basename(rel))
    out_name = stem + suffix
    if rel_dir == "":
        return os.path.join(output_dir, out_name)
    return os.path.join(output_dir, rel_dir, out_name)


def save_binary_mask(mask: np.ndarray, save_path: str) -> None:
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(save_path)


def main() -> int:
    args = parse_args()
    args.device = choose_device(args.device)

    if not os.path.exists(args.input_dir) or not os.path.isdir(args.input_dir):
        print(f"[ERROR] 输入目录不存在或不是文件夹: {args.input_dir}", file=sys.stderr)
        return 1

    if not os.path.exists(args.checkpoint_path) or not os.path.isfile(args.checkpoint_path):
        print(f"[ERROR] checkpoint 不存在或不是文件: {args.checkpoint_path}", file=sys.stderr)
        return 1

    images = list_images(args.input_dir)
    if not images:
        print(f"[ERROR] 在 {args.input_dir} 下未找到图片文件。", file=sys.stderr)
        return 1

    bpe_path = resolve_bpe_path(args.bpe_path)

    print(f"[INFO] device = {args.device}")
    print(f"[INFO] checkpoint_path = {args.checkpoint_path}")
    print(f"[INFO] bpe_path = {bpe_path if bpe_path is not None else 'None'}")
    print(f"[INFO] prompt = {args.prompt}")
    print(f"[INFO] 共找到 {len(images)} 张图片")

    try:
        segmenter = LocalSam3Segmenter(
            checkpoint_path=os.path.abspath(args.checkpoint_path),
            bpe_path=bpe_path,
            device=args.device,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
        )
    except Exception as e:
        print(f"[ERROR] SAM3 本地加载失败: {type(e).__name__}: {e}", file=sys.stderr)
        if bpe_path is None:
            print(
                f"[HINT] 当前未自动找到 {BPE_FILENAME}，如你的安装环境没有内置该文件，请显式传入 --bpe_path",
                file=sys.stderr,
            )
        return 1

    success = 0
    failed = 0

    for idx, image_path in enumerate(images, start=1):
        out_path = output_path_for_image(
            image_path=image_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            suffix=args.suffix,
        )

        if args.skip_existing and os.path.exists(out_path):
            print(f"[{idx}/{len(images)}] SKIP {out_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            mask = segmenter.predict_union_mask(image=image, prompt=args.prompt)
            save_binary_mask(mask, out_path)
            success += 1
            print(f"[{idx}/{len(images)}] OK   {image_path} -> {out_path}")
        except Exception as e:
            failed += 1
            print(f"[{idx}/{len(images)}] FAIL {image_path}: {type(e).__name__}: {e}", file=sys.stderr)
            if args.fail_fast:
                raise

    print(f"[DONE] success={success}, failed={failed}, output_dir={args.output_dir}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())