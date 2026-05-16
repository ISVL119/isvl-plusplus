import os
import argparse
import random
import gc

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from segment_anything_hq import sam_model_registry, SamPredictor


_CLASS_NAMES_ = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_topn_bounding_boxes(binary_image, top_n=3):
    binary_image = (binary_image > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

    if num_labels <= 1:
        return []

    areas = stats[1:, cv2.CC_STAT_AREA]
    top_indices = np.argsort(areas)[-top_n:] + 1

    bounding_boxes = []
    for idx in top_indices:
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        bounding_boxes.append([x, y, x + w, y + h])

    return bounding_boxes


def fill_holes(image):
    image = image.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(image)
    mask = np.zeros_like(image)

    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if contours:
            cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    filled_image = cv2.bitwise_or(image, mask)
    return filled_image


def tensor_to_numpy(x):
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def normalize_hqsam_masks(masks, target_shape):
    masks = tensor_to_numpy(masks)
    target_h, target_w = target_shape

    if masks is None:
        return np.zeros((0, target_h, target_w), dtype=bool)

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None]

    if masks.ndim != 3:
        return np.zeros((0, target_h, target_w), dtype=bool)

    normalized = []

    for mask in masks:
        mask = mask.astype(np.float32)

        if mask.shape != target_shape:
            mask = cv2.resize(
                mask,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )

        normalized.append(mask > 0.5)

    if len(normalized) == 0:
        return np.zeros((0, target_h, target_w), dtype=bool)

    return np.stack(normalized, axis=0)


def select_hqsam_mask(masks, scores, coarse_mask, mode="union"):
    if masks.shape[0] == 0:
        return np.zeros_like(coarse_mask, dtype=np.uint8)

    scores = tensor_to_numpy(scores)
    coarse = coarse_mask > 0

    if mode == "union":
        selected = masks.max(axis=0)

    elif mode == "score" and scores is not None and len(scores) == masks.shape[0]:
        selected = masks[int(np.argmax(scores))]

    elif mode == "coarse_iou":
        best_idx = 0
        best_iou = -1.0

        for idx, mask in enumerate(masks):
            inter = np.logical_and(mask, coarse).sum()
            union = np.logical_or(mask, coarse).sum()
            iou = inter / (union + 1e-6)

            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        selected = masks[best_idx]

    else:
        selected = masks[0]

    return selected.astype(np.uint8) * 255


def apply_coarse_constraint(mask, coarse_mask, kernel_size=15, iterations=1):
    if kernel_size <= 0:
        return mask

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    allowed = cv2.dilate(
        (coarse_mask > 0).astype(np.uint8),
        kernel,
        iterations=iterations,
    )

    constrained = ((mask > 0) & (allowed > 0)).astype(np.uint8) * 255
    return constrained


def read_rgb_image_resized(image_path, target_shape):
    target_h, target_w = target_shape[:2]
    image = Image.open(image_path).convert("RGB")

    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR

    if image.size != (target_w, target_h):
        image = image.resize((target_w, target_h), resample=resample)

    return image


def pil_to_numpy_rgb(image):
    return np.asarray(image)


def build_hqsam_predictor(args, device):
    sam = sam_model_registry[args.hqsam_model_type](checkpoint=args.hqsam_checkpoint)
    sam = sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)
    return predictor


def predict_hqsam_box_mask(
    predictor,
    bbox,
    binary_mask,
    multimask_output=False,
    select_mode="union",
    constraint_kernel=15,
    constraint_iterations=1,
):
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            box=np.array(bbox),
            multimask_output=multimask_output,
            return_logits=False,
        )

    masks = normalize_hqsam_masks(masks, binary_mask.shape)

    mask = select_hqsam_mask(
        masks=masks,
        scores=scores,
        coarse_mask=binary_mask,
        mode=select_mode,
    )

    mask = apply_coarse_constraint(
        mask=mask,
        coarse_mask=binary_mask,
        kernel_size=constraint_kernel,
        iterations=constraint_iterations,
    )

    return mask


def maybe_print_cuda_memory(args, class_name, image_idx):
    if not torch.cuda.is_available():
        return

    if args.print_memory_interval <= 0:
        return

    if image_idx % args.print_memory_interval != 0:
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(
        f"[GPU][{class_name}][{image_idx}] "
        f"allocated={allocated:.2f}GB, reserved={reserved:.2f}GB"
    )


def maybe_clear_cuda_cache(args, image_idx):
    if not torch.cuda.is_available():
        return

    if args.empty_cache_interval <= 0:
        return

    if image_idx % args.empty_cache_interval != 0:
        return

    torch.cuda.empty_cache()


def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True

    if value in ("no", "false", "f", "0", "n"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def samfiner(args, test_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_hqsam_predictor(args, device)

    dataset_dir = args.data_path
    save_dir = args.bin_savedir

    if test_type == "test_public":
        binary_dir = os.path.join(args.bin_savedir, "anomaly_images_thresholded_public")
        save_dir = os.path.join(save_dir, "anomaly_images_thresholded_public")
    else:
        binary_dir = os.path.join(args.bin_savedir, "anomaly_images_thresholded")
        save_dir = os.path.join(save_dir, "anomaly_images_thresholded")

    class_names = _CLASS_NAMES_ if args.class_name == "all" else [args.class_name]

    for class_name in class_names:
        print(class_name)
        setup_seed(1)

        dataset_class_dir = os.path.join(dataset_dir, class_name, test_type)
        image_path_list = []

        if test_type == "test_public":
            image_name_list = os.listdir(os.path.join(dataset_class_dir, "bad"))
            image_path_list.extend(
                [
                    os.path.join(dataset_class_dir, "bad", image_name)
                    for image_name in image_name_list
                ]
            )

            image_name_list = os.listdir(os.path.join(dataset_class_dir, "good"))
            image_path_list.extend(
                [
                    os.path.join(dataset_class_dir, "good", image_name)
                    for image_name in image_name_list
                ]
            )

        else:
            image_name_list = os.listdir(dataset_class_dir)
            image_path_list.extend(
                [
                    os.path.join(dataset_class_dir, image_name)
                    for image_name in image_name_list
                ]
            )

        image_path_list = sorted(image_path_list)

        for image_idx, image_path in enumerate(tqdm(image_path_list)):
            binary_mask_path = image_path.replace(dataset_dir, binary_dir)
            binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

            if binary_mask is None:
                raise FileNotFoundError(f"Cannot read binary mask: {binary_mask_path}")

            binary_mask = (binary_mask > 0).astype(np.uint8) * 255
            kernel = np.ones((3, 3), dtype=np.uint8)   # 膨胀 3 个像素 => 2*3+1=7
            prompt_mask = cv2.dilate((binary_mask > 0).astype(np.uint8), kernel, iterations=1) * 255
            bbox_list = get_topn_bounding_boxes(prompt_mask, top_n=args.top_n)

            if len(bbox_list) == 0:
                masks = np.zeros_like(binary_mask, dtype=np.uint8)

            else:
                image = read_rgb_image_resized(image_path, binary_mask.shape)
                image_np = pil_to_numpy_rgb(image)

                with torch.inference_mode():
                    predictor.set_image(image_np)

                mask_list = []
                for bbox in bbox_list:
                    mask = predict_hqsam_box_mask(
                        predictor=predictor,
                        bbox=bbox,
                        binary_mask=binary_mask,
                        multimask_output=args.multimask_output,
                        select_mode=args.mask_select,
                        constraint_kernel=args.constraint_kernel,
                        constraint_iterations=args.constraint_iterations,
                    )
                    mask_list.append(mask)

                masks = np.array(mask_list).max(0).astype(np.uint8)

                if args.fill_holes:
                    masks = fill_holes(masks)

                del image
                del image_np
                del mask_list

            save_path = binary_mask_path.replace(binary_dir, save_dir)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, masks)

            del binary_mask
            del bbox_list
            del masks

            gc.collect()
            maybe_print_cuda_memory(args, class_name, image_idx)
            maybe_clear_cuda_cache(args, image_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--data_path",
        type=str,
        default=r"./datasets/mvtec_ad_2",
    )
    parser.add_argument(
        "--bin_savedir",
        type=str,
        default=r"./results/",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        default=r"challenge",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="all",
        choices=["all"] + _CLASS_NAMES_,
    )
    parser.add_argument(
        "--hqsam_checkpoint",
        type=str,
        default="backbones/weights/sam_hq_vit_h.pth",
    )
    parser.add_argument(
        "--hqsam_model_type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b", "vit_tiny"],
    )
    parser.add_argument(
        "--multimask_output",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--mask_select",
        type=str,
        default="union",
        choices=["coarse_iou", "score", "union"],
    )
    parser.add_argument(
        "--constraint_kernel",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--constraint_iterations",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--fill_holes",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--empty_cache_interval",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--print_memory_interval",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if args.test_type != "challenge":
        samfiner(args, test_type="test_public")
    else:
        samfiner(args, test_type="test_private")
        samfiner(args, test_type="test_private_mixed")