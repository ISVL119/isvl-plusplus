import os
import random
import hashlib
import numpy as np
from PIL import Image
import cv2
import argparse
from pathlib import Path
import torch



IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(
        description="generate synthesized data by hand with normal images, anomalous patches and foreground"
    )
    p.add_argument("--anomaly_dir", default='./data/anomaly', type=str,
                   help="path for anomaly patches")
    p.add_argument("--normal_dir", default='../datasets/mvtec_ad_2', type=str,
                   help="normal background images")
    p.add_argument("--mask_dir", type=str, default='./log/foreground/foreground_mvtecad2_keepsize_sam3',
                   help="foreground png masks for normal images")
    p.add_argument("--output_dir", default='./log/synthesized/synthesized_mvtecad2_1024rgbl', type=str,
                   help="path for final INP++ style output synthesized images")
    p.add_argument("--dataset-name", type=str, default="mvtec_ad_2")
    p.add_argument("--num_per_category", type=int, default=1024,
                   help="每个类别总共生成的合成图片数量")
    p.add_argument("--seed", type=int, default=6,
                   help="随机种子，确保可复现")
    p.add_argument("--category", type=str, nargs="+", default=["can", "fabric", "rice", "vial", "fruit_jelly", "sheet_metal", "wallplugs", "walnuts"],
                   help="要处理的类别名称，可一次输入多个类别，例如 --category vial bottle cable")

    p.add_argument("--output_size", type=int, default=640,
                   help="最终保存的合成图和mask尺寸；异常合成仍然在原图尺寸上完成")
    return p.parse_args()


def stable_seed(*items):
    text = "::".join(str(x) for x in items)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2 ** 32)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)


def _list_patch_images(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda x: str(x)
    )


def _list_normal_images(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda x: str(x)
    )


def _collect_leaf_patch_dirs(anomaly_root: Path):
    patch_dirs = []

    if anomaly_root.is_dir() and len(_list_patch_images(anomaly_root)) > 0:
        patch_dirs.append(anomaly_root)

    for p in anomaly_root.rglob("*"):
        if p.is_dir() and len(_list_patch_images(p)) > 0:
            patch_dirs.append(p)

    return sorted(set(patch_dirs), key=lambda x: str(x))


def _clear_image_files(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)

    for p in sorted(folder.iterdir(), key=lambda x: str(x)):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            p.unlink()


def _is_texture_category(cat_name: str):
    texture_categories = {
        "carpet", "grid", "leather", "tile", "wood"
    }
    return cat_name.lower() in texture_categories or "texture" in cat_name.lower()


def get_patch_meta_from_leaf_dir(patch_dir: Path, anomaly_root: Path):
    rel_parts = patch_dir.relative_to(anomaly_root).parts

    if len(rel_parts) >= 2:
        anomaly_shape_name = rel_parts[-2]
        patch_source_name = rel_parts[-1]
    elif len(rel_parts) == 1:
        anomaly_shape_name = rel_parts[-1]
        patch_source_name = rel_parts[-1]
    else:
        anomaly_shape_name = patch_dir.name
        patch_source_name = patch_dir.name

    return anomaly_shape_name, patch_source_name


def read_foreground_png(mask_path: Path, target_hw):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"cannot read foreground png: {mask_path}")

    target_h, target_w = target_hw

    if mask.shape != (target_h, target_w):
        print(
            f"[WARN] mask shape {mask.shape} does not match image shape {(target_h, target_w)}: "
            f"{mask_path}, resize mask to original image size"
        )
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    mask = (mask > 0).astype(np.float32)
    return mask


def random_anomaly_patch(anom_img: Image.Image, rng: random.Random, min_scale=0.005, max_scale=0.01):
    w, h = anom_img.size
    scale = rng.uniform(min_scale, max_scale)
    new_w, new_h = int(w * scale), int(h * scale)
    new_w = max(new_w, 1)
    new_h = max(new_h, 1)

    patch = anom_img.resize((new_w, new_h), Image.NEAREST)

    angle = rng.uniform(0, 360)
    patch = patch.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)

    return patch


def overlay_patch_on_image(base_img: Image.Image,
                           fg_mask: np.ndarray,
                           patch: Image.Image,
                           rng: random.Random):
    bw, bh = base_img.size
    pw, ph = patch.size

    if pw > bw or ph > bh:
        return base_img, np.zeros((bh, bw), dtype=np.uint8)

    ys, xs = np.where(fg_mask >= 0.8)
    if len(xs) == 0:
        return base_img, np.zeros((bh, bw), dtype=np.uint8)

    idx = rng.randrange(len(xs))
    cx, cy = xs[idx], ys[idx]

    x1 = max(0, cx - pw // 2)
    y1 = max(0, cy - ph // 2)
    x2 = min(bw, x1 + pw)
    y2 = min(bh, y1 + ph)
    w, h = x2 - x1, y2 - y1

    if w <= 0 or h <= 0:
        return base_img, np.zeros((bh, bw), dtype=np.uint8)

    src_x = max(0, x1 - (cx - pw // 2))
    src_y = max(0, y1 - (cy - ph // 2))

    patch_np = np.array(patch)
    if patch.mode == 'RGBA':
        raw_alpha = patch_np[:, :, 3]
        alpha_patch = (raw_alpha > 0).astype(float)
        rgb_patch = patch_np[:, :, :3].astype(float)
    else:
        gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
        alpha_patch = (gray < 250).astype(float)
        rgb_patch = patch_np.astype(float)

    alpha_crop = alpha_patch[src_y:src_y + h, src_x:src_x + w]
    rgb_crop = rgb_patch[src_y:src_y + h, src_x:src_x + w, :]

    sub_mask = fg_mask[y1:y1 + h, x1:x1 + w].astype(float)

    h_min = min(alpha_crop.shape[0], sub_mask.shape[0])
    w_min = min(alpha_crop.shape[1], sub_mask.shape[1])
    alpha_crop = alpha_crop[:h_min, :w_min]
    sub_mask = sub_mask[:h_min, :w_min]
    rgb_crop = rgb_crop[:h_min, :w_min, :]

    sub_mask_binary = np.where(sub_mask > 0.45, 1.0, 0.0)

    valid_crop = alpha_crop * sub_mask
    valid_crop_mask = (alpha_crop * sub_mask_binary).astype(np.uint8)

    valid_mask = np.zeros((bh, bw), dtype=np.uint8)
    valid_mask[y1:y1 + h_min, x1:x1 + w_min] = valid_crop_mask

    base_rgb = base_img.convert('RGB')
    base_np = np.array(base_rgb).astype(float)


    base_np = np.array(base_img).astype(float)
    if base_img.mode == 'L':
        patch_gray = cv2.cvtColor(rgb_crop.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(float)
        base_np[y1:y1 + h_min, x1:x1 + w_min] = (
            valid_crop * patch_gray +
            (1 - valid_crop) * base_np[y1:y1 + h_min, x1:x1 + w_min]
        )
        return Image.fromarray(base_np.astype(np.uint8), mode='L'), valid_mask

    for c in range(3):
        base_np[y1:y1 + h_min, x1:x1 + w_min, c] = (
            valid_crop * rgb_crop[:, :, c] +
            (1 - valid_crop) * base_np[y1:y1 + h_min, x1:x1 + w_min, c]
        )

    return Image.fromarray(base_np.astype(np.uint8), mode='RGB'), valid_mask




def read_image(path, resize=None):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize:
        img = cv2.resize(img, dsize=resize)
    return img


def _apply_patch_transform(patch: Image.Image, patch_source_name: str, rng: random.Random):
    scale_factor = 2.7

    if patch_source_name == 'bottle' or patch_source_name == 'capsules':
        return random_anomaly_patch(patch, rng, min_scale=0.1 * scale_factor, max_scale=1.0 * scale_factor)
    elif patch_source_name == 'contamination':
        return random_anomaly_patch(patch, rng, min_scale=0.1 * scale_factor, max_scale=0.8 * scale_factor)
    elif patch_source_name == 'tube':
        return random_anomaly_patch(patch, rng, min_scale=0.05 * scale_factor, max_scale=0.3 * scale_factor)
    elif patch_source_name == 'hair':
        return random_anomaly_patch(patch, rng, min_scale=0.05 * scale_factor, max_scale=0.2 * scale_factor)
    elif patch_source_name == 'screw':
        return random_anomaly_patch(patch, rng, min_scale=0.05 * scale_factor, max_scale=0.1 * scale_factor)
    elif patch_source_name == 'nut':
        return random_anomaly_patch(patch, rng, min_scale=0.05 * scale_factor, max_scale=0.2 * scale_factor)
    elif patch_source_name == 'blocky_defect':
        return random_anomaly_patch(patch, rng, min_scale=0.05 * scale_factor, max_scale=0.25 * scale_factor)
    elif patch_source_name == 'Linear_defect':
        return random_anomaly_patch(patch, rng, min_scale=0.05 * scale_factor, max_scale=0.5 * scale_factor)
    else:
        return random_anomaly_patch(patch, rng, min_scale=0.05 * scale_factor, max_scale=0.2 * scale_factor)


args = parse_args()

os.environ['PYTHONHASHSEED'] = str(args.seed)
seed_all(args.seed)
torch.use_deterministic_algorithms(True)


normal_root = Path(args.normal_dir)
anomaly_root = Path(args.anomaly_dir)
mask_root = Path(args.mask_dir)

if args.output_dir is None:
    args.output_dir = f'log/synthesized/synthesized_{args.dataset_name}_inp++'

output_root = Path(args.output_dir)
output_root.mkdir(parents=True, exist_ok=True)

patch_dirs = _collect_leaf_patch_dirs(anomaly_root)

if len(patch_dirs) == 0:
    raise RuntimeError(f"no anomaly patch dirs found in {anomaly_root}")

patch_candidates_by_dir = {}
for patch_dir in patch_dirs:
    patch_candidates = _list_patch_images(patch_dir)
    if len(patch_candidates) > 0:
        patch_candidates_by_dir[patch_dir] = patch_candidates

if len(patch_candidates_by_dir) == 0:
    raise RuntimeError(f"no anomaly patch images found in {anomaly_root}")

patch_dirs_ordered = sorted(patch_candidates_by_dir.keys(), key=lambda x: str(x))

all_cats = sorted([d for d in normal_root.iterdir() if d.is_dir()], key=lambda x: str(x))

for idx, category_dir in enumerate(all_cats):
    cat_name = category_dir.name

    if args.category is not None and cat_name not in args.category:
        continue

    category_rng = random.Random(stable_seed(args.seed, cat_name))

    category_out_dir = output_root / cat_name
    category_image_dir = category_out_dir / "images"
    category_mask_dir = category_out_dir / "masks"
    _clear_image_files(category_image_dir)
    _clear_image_files(category_mask_dir)

    norm_good_dir = category_dir / "train" / "good"
    if not norm_good_dir.exists():
        print(f"[WARN] no normal train/good dir for {cat_name}, skipping")
        continue

    normal_candidates = _list_normal_images(norm_good_dir)
    if len(normal_candidates) == 0:
        print(f"[WARN] no normal images found for {cat_name}, skipping")
        continue

    mask_good_dir = mask_root / cat_name / "train" / "good"
    use_full_mask = False

    if not mask_good_dir.exists():
        if _is_texture_category(cat_name):
            use_full_mask = True
            print(f"[INFO] no mask train/good dir for texture category '{cat_name}', use full-image mask")
        else:
            print(f"[WARN] no mask train/good dir for {cat_name}, skipping")
            continue

    generated_count = 0
    attempt_count = 0
    max_attempt_count = args.num_per_category * 20

    while generated_count < args.num_per_category and attempt_count < max_attempt_count:
        attempt_count += 1

        norm_path = category_rng.choice(normal_candidates)
        norm_name = norm_path.stem

        with Image.open(norm_path) as raw_img:
            if raw_img.mode == 'L':
                img = raw_img.convert('L')
            else:
                img = raw_img.convert('RGB')

        orig_w, orig_h = img.size

        if use_full_mask:
            mask = np.ones((orig_h, orig_w), dtype=np.float32)
        else:
            mask_path = mask_good_dir / f"f_{norm_name}.png"
            if not mask_path.exists():
                if _is_texture_category(cat_name):
                    mask = np.ones((orig_h, orig_w), dtype=np.float32)
                    print(f"[INFO] no png mask file for texture image '{norm_path.name}', use full-image mask")
                else:
                    print(f"[WARN] png mask file not found: {mask_path}, skipping {norm_path.name}")
                    continue
            else:
                try:
                    mask = read_foreground_png(mask_path, target_hw=(orig_h, orig_w))
                except Exception as e:
                    print(f"[WARN] invalid png mask: {mask_path}, {e}, skipping {norm_path.name}")
                    continue

        patch_dir = category_rng.choice(patch_dirs_ordered)
        patch_path = category_rng.choice(patch_candidates_by_dir[patch_dir])
        anomaly_shape_name, patch_source_name = get_patch_meta_from_leaf_dir(patch_dir, anomaly_root)

        with Image.open(patch_path) as patch_raw:
            patch = patch_raw.convert('RGBA')

        patch = _apply_patch_transform(patch, patch_source_name, category_rng)

        out_img, valid_mask = overlay_patch_on_image(img, mask, patch, category_rng)

        if valid_mask.sum() == 0:
            continue

        output_size = (args.output_size, args.output_size)
        out_img = out_img.resize(output_size, Image.Resampling.BILINEAR)

        out_name = f"{cat_name}_{norm_name}_syn_{anomaly_shape_name}_{patch_source_name}_{generated_count}.png"
        out_img.save(category_image_dir / out_name, compress_level=0)

        bin_mask = (valid_mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(bin_mask).resize(output_size, Image.Resampling.NEAREST)
        mask_img.save(category_mask_dir / out_name, compress_level=0)

        generated_count += 1

    if generated_count < args.num_per_category:
        print(f"[WARN] only generated {generated_count}/{args.num_per_category} images for {cat_name}")

    print(
        f"[OK] {cat_name}: "
        f"images={generated_count}, "
        f"masks={generated_count}, "
        f"saved_to={category_out_dir}"
    )

print("合成完成，最终结果保存在：", output_root)