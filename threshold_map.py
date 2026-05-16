import os
import json
import cv2
from decimal import Decimal, ROUND_HALF_UP


def load_thresholds_from_json(json_path):
    """
    从本地JSON文件读取阈值，并对数值进行四舍五入后返回。
    JSON结构示例：
    {
        "rice": {
            "test_private": 15.50923676617176,
            "test_private_mixed": 18.237899064206832
        },
        ...
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw_thresholds = json.load(f)

    thresholds = {}
    for category, category_thresholds in raw_thresholds.items():
        thresholds[category] = {}
        for subfolder, value in category_thresholds.items():
            rounded_value = int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            thresholds[category][subfolder] = rounded_value

    return thresholds


def threshold_and_save_images_recursive(input_dir, output_dir, thresholds):
    """
    递归地将input_dir下的所有单通道图片按类别+子文件夹对应的阈值进行二值化，
    并保留原目录结构保存到output_dir，统一保存为PNG格式。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(root, file)

                # 计算相对路径
                rel_path = os.path.relpath(file_path, input_dir)
                rel_parts = rel_path.split(os.sep)

                # 确保至少有类别和子目录
                if len(rel_parts) < 2:
                    print(f"Warning: File {file_path} not in expected category/subfolder structure.")
                    continue

                category = rel_parts[0]
                subfolder = rel_parts[1]

                # 获取阈值
                if category in thresholds:
                    category_thresholds = thresholds[category]
                    if subfolder in category_thresholds:
                        thresh = category_thresholds[subfolder]
                    elif "default" in category_thresholds:
                        thresh = category_thresholds["default"]
                        print(f"Info: Using default threshold for {category}/{subfolder}")
                    else:
                        print(f"Warning: No threshold found for {category}/{subfolder}, skipping.")
                        continue
                else:
                    print(f"Warning: Category '{category}' not in thresholds, skipping {file_path}")
                    continue

                # 读取图像
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to read image: {file_path}")
                    continue

                # 阈值处理
                _, binary_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

                # 保存路径
                base_filename = os.path.splitext(file)[0] + ".png"
                save_dir = os.path.join(output_dir, os.path.dirname(rel_path))
                save_path = os.path.join(save_dir, base_filename)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                cv2.imwrite(save_path, binary_img)


if __name__ == "__main__":
    input_folder = "./results/anomaly_images"  # TODO: 替换为你的输入路径
    output_folder = "./results/anomaly_images_thresholded"  # TODO: 替换为你的输出路径
    threshold_json_path = "./final_thresholds_by_split.json"  # TODO: 替换为你的JSON路径

    thresholds = load_thresholds_from_json(threshold_json_path)
    threshold_and_save_images_recursive(input_folder, output_folder, thresholds)