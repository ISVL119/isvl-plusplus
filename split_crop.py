import os
import cv2
import numpy as np
import argparse
import math


def get_start_positions(length, window_size, desired_overlap):
    if length <= window_size:
        return [0]

    if length % window_size == 0:
        return list(range(0, length - window_size + 1, window_size))

    nominal_step = int(window_size * (1 - desired_overlap))
    nominal_step = max(nominal_step, 1)

    last_step = length - window_size
    num_windows = int(math.ceil(last_step / nominal_step)) + 1

    steps = np.linspace(0, last_step, num_windows)
    steps = np.round(steps).astype(int).tolist()
    steps = sorted(set(steps))

    return steps


def crop_window_with_boundary(img, x, y, window_size):
    height, width = img.shape[:2]

    window_x1 = min(x, max(width - window_size, 0))
    window_y1 = min(y, max(height - window_size, 0))
    window_x2 = min(window_x1 + window_size, width)
    window_y2 = min(window_y1 + window_size, height)

    window = img[window_y1:window_y2, window_x1:window_x2]

    if window.shape[0] < window_size or window.shape[1] < window_size:
        pad_bottom = window_size - window.shape[0]
        pad_right = window_size - window.shape[1]
        window = cv2.copyMakeBorder(
            window,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_REPLICATE,
        )

    return window, window_x1, window_y1, window_x2, window_y2


def generate_sliding_window_images(img_path, img_crop_path, window_size, desired_overlap):
    img_name = os.path.basename(img_path).split(".")[0]

    img = cv2.imread(img_path)

    if img is None:
        print(f"Failed to read image: {img_path}")
        return

    height, width, _ = img.shape

    y_steps = get_start_positions(height, window_size, desired_overlap)
    x_steps = get_start_positions(width, window_size, desired_overlap)

    for y in y_steps:
        for x in x_steps:
            window, x1, y1, x2, y2 = crop_window_with_boundary(img, x, y, window_size)

            file_path = os.path.join(
                img_crop_path,
                f"{img_name}_x{x1}_y{y1}_x{x2 - 1}_y{y2 - 1}.png"
            )

            cv2.imwrite(file_path, window)


def process_images(img_files, img_dir, crop_dir, window_size, desired_overlap):
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        generate_sliding_window_images(img_path, crop_dir, window_size, desired_overlap)


def crop(path, crop_path, window_size, desired_overlap, class_name=None):
    classname_list = ['sheet_metal', 'vial', 'wallplugs', 'walnuts', 'can', 'fabric', 'fruit_jelly', 'rice']

    if class_name is None or len(class_name) == 0:
        selected_classname_list = classname_list
    elif isinstance(class_name, str):
        selected_classname_list = [class_name]
    else:
        selected_classname_list = class_name

    for ct in selected_classname_list:
        if ct not in classname_list:
            print(f"Skip unknown class: {ct}")
            continue

        print(f"{ct} processing...")

        if not os.path.isdir(os.path.join(path, ct)):
            continue

        ct_path = os.path.join(path, ct)
        cp_path = os.path.join(crop_path, ct)

        for category in os.listdir(ct_path):
            category_path = os.path.join(ct_path, category)
            crop_path_1 = os.path.join(cp_path, category)

            if category in ['test_private', 'test_private_mixed']:
                img_files = os.listdir(category_path)
                process_images(img_files, category_path, crop_path_1, window_size, desired_overlap)
            else:
                for label in os.listdir(category_path):
                    label_path = os.path.join(category_path, label)
                    crop_path_2 = os.path.join(crop_path_1, label)

                    if label == 'ground_truth':
                        for gt in os.listdir(label_path):
                            gt_path = os.path.join(label_path, gt)
                            crop_path_3 = os.path.join(crop_path_2, gt)
                            img_files = os.listdir(gt_path)
                            process_images(img_files, gt_path, crop_path_3, window_size, desired_overlap)
                    else:
                        img_files = os.listdir(label_path)
                        process_images(img_files, label_path, crop_path_2, window_size, desired_overlap)

        print(f"{ct} finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data_path', type=str, default=r'./datasets/mvtec_ad_2')
    parser.add_argument('--save_path', type=str, default=r'./datasets/mvtec_ad_2_splits_1024')
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--desired_overlap', type=float, default=0.2)
    parser.add_argument(
    '--class_name',
    type=str,
    nargs='*',
    default=['sheet_metal', 'wallplugs', 'walnuts', 'can', 'fabric', 'fruit_jelly', 'rice', 'vial']
    )

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    for class_name in args.class_name:
        crop(
            path=args.data_path,
            crop_path=args.save_path,
            window_size=args.window_size,
            desired_overlap=args.desired_overlap,
            class_name=[class_name],
        )