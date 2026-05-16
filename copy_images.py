import os
import argparse
import shutil


def copy_train_good_images(data_path, save_path, class_name=None):
    classname_list = [
        'sheet_metal',
        'vial',
        'wallplugs',
        'walnuts',
        'can',
        'fabric',
        'fruit_jelly',
        'rice',
    ]

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

        src_dir = os.path.join(data_path, ct, 'train', 'good')
        dst_dir = os.path.join(save_path, ct, 'train', 'good')

        if not os.path.isdir(src_dir):
            print(f"Skip missing dir: {src_dir}")
            continue

        os.makedirs(dst_dir, exist_ok=True)

        for img_file in os.listdir(src_dir):
            src_path = os.path.join(src_dir, img_file)
            dst_path = os.path.join(dst_dir, img_file)

            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)

        print(f"{ct} train/good copied.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy train/good images to cropped dataset directory.')

    parser.add_argument('--data_path', type=str, default=r'./datasets/mvtec_ad_2')
    parser.add_argument('--save_path', type=str, default=r'./datasets/mvtec_ad_2_splits_1024')
    parser.add_argument(
        '--class_name',
        type=str,
        nargs='*',
        default=['sheet_metal', 'wallplugs', 'walnuts', 'can', 'fabric', 'fruit_jelly', 'rice', 'vial']
    )

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    copy_train_good_images(
        data_path=args.data_path,
        save_path=args.save_path,
        class_name=args.class_name,
    )