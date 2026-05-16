# ISVL++

We are releasing the paper first. The code will be made public after cleanup, approximately one day later, and no later than required by the competition rules, i.e., within three days after the submission deadline.

Update(2026.5.26): We have released all the code 
## Install Environment

Create a new conda environment and install the required dependencies:

```bash
# create a new conda environment
conda create -n cvprw python=3.11.13

# activate the environment
conda activate cvprw

# install torch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# install other dependencies
pip install -r requirements.txt
```

> **Note**
> All experiments were conducted on an NVIDIA GeForce RTX 4090 (24 GB). For the best reproducibility, we recommend using the same GPU and package versions. :contentReference[oaicite:0]{index=0}

## Additional Notes on Disk Usage

The entire project folder may temporarily occupy approximately **160 GB** of disk space, including the original compressed dataset files (about **30 GB**).

## Dataset Preparation

This project uses the following datasets:

- **MVTec AD 2** for training and evaluation
- **DTD (Describable Textures Dataset)** for validation threshold selection
- **Anomaly Bank** in `./sam3-260512/data/anomaly` for anomaly synthesis during training

### 1. Prepare MVTec AD 2

Extract the **MVTec AD 2** dataset to:

```bash
./datasets/mvtec_ad_2
```

### 2. Prepare DTD Dataset

Download the **[DTD dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)** from the official website, then extract it to:

```bash
./datasets/dtd
```

Please make sure all datasets are placed in the correct directories before training or evaluation.

## Pre-trained Weights Preparation

We use **DINOv3** as the backbone for feature extraction, **SAM 3** and **DenseNet** for foreground extraction, and **HQ-SAM** for post-processing.

### Weight Overview

- **DINOv3**
  - Model: `dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`
  - Download: [DINOv3 weights](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)

- **SAM 3**
  - Download: [facebook/sam3](https://huggingface.co/facebook/sam3/tree/main)

- **HQ-SAM**
  - Model: `sam_hq_vit_h.pth`
  - Download: [HQ-SAM weights](https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_h.pth)

- **DenseNet**
  - DenseNet weights will be downloaded automatically from Hugging Face during execution.

Please place all manually downloaded weights in:

```bash
./backbones/weights
```

## Reproduce the Results

Please make sure the project directory is organized correctly and that both `mvtec_ad_2` and `dtd` are placed under `./datasets` and all weights in correct path.

The directory structure should look like this:

```text
ISVL++
├── beit
├── datasets
│   ├── mvtec_ad_2
│   │   ├── can
│   │   ├── fabric
│   │   └── ...
│   └── dtd
│       └── images
├── backbones
│   └── weights
│       ├── dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
│       ├── sam_hq_vit_h.pth
│       └── sam3.pt
```
### Option 1: Start from Model Inference

If you only want to quickly verify the results, we provide trained checkpoint files here:

[Google Drive Checkpoints](https://drive.google.com/drive/folders/1BWTVDGPyyhErIhsY14CjJGculIvsmrdJ?usp=sharing)

Please place the downloaded files into:

```bash
./saved_results/INP-Former-Multi-Class_dataset=Mvtec_ad_2_Encoder=dinov3_vith16plus_Resize=512_INP_num=6_Seg=512
```

Since the model is evaluated and tested on cropped images, you need to first split the images by running:

```bash
python split_crop.py
```

Then start from **Step 4: Model Inference** in `reproduce_result.sh`.

After the script finishes, the results will be generated in the current working directory.

### Option 2: Run the Full Reproduction Script

You can also directly run:

```bash
bash reproduce_result.sh
```

The final submission file will be saved as:

```bash
results.tar.gz
```

## Step-by-step Explanation

### 1. Dataset Processing

We first split the original **MVTec AD 2** images into **1024 × 1024** patches, and then copy the original full-resolution training images into the processed dataset so that the model can leverage both local details and global context during training.

```bash
python split_crop.py
python copy_images.py
```

### 2. Anomaly Synthesis

We observe that structural anomalies often share similar visual patterns, so we build two anomaly libraries, **`blocky_defect`** and **`linear_defect`**. We use **SAM 3** for foreground extraction, and if SAM 3 fails and produces an all-black mask, we fall back to the foreground extraction strategy from **CPR**. The anomaly regions are then randomly pasted onto training images to generate synthetic anomalies for model training.

```bash
cd sam3-260512
python 1_foreground_extraction.py
python 2_generate_synthetic_anomaly.py
mv ./log/synthesized/synthesized_mvtecad2_1024rgbl ../datasets/
cd ../
```

### 3. Model Training

We adopt a one-class-per-model training strategy, where a separate model is trained for each category.

```bash
# example: train the model for the "can" category
python isvl_train_and_test++.py --use_synth_anomalies --item_list can --phase train --synth_anomaly_root ./datasets/synthesized_mvtecad2_1024rgbl/can
```

### 4. Model Inference

We perform inference on cropped image patches, and the pixel values in overlapping regions are averaged when merging the patch-level predictions into the final anomaly map.

```bash
python isvl_train_and_test++.py --item_list can --phase test
```

### 5. Threshold Selection

We start from a statistical thresholding strategy based on **mean + k × standard deviation** on the validation set, and reformulate threshold selection as the problem of finding the optimal **k**. We then tune **k** on the validation set using the **DTD** dataset together with anomaly synthesis and illumination augmentation. The selected thresholds will be saved to `final_thresholds_by_split.json` in the current working directory.

```bash
python isvl_select_threshold.py --item_list can --phase true_val --synthetic_k_enable --illumination_calibration_enable
```

### 6. Binarization and Post-processing

We first read the selected thresholds from `final_thresholds_by_split.json` to binarize the anomaly maps, and then follow the idea of **SAM-Finer** from **[RoBiS](https://github.com/xrli-U/RoBiS)** by replacing SAM with **HQ-SAM** to obtain more refined segmentation results.

```bash
python threshold_map.py
python SAMHQ-Finer.py
```

### 7. Submission Preparation

We first convert the predicted anomaly maps to **16-bit** format, then verify and package the results for submission. The final upload file will be saved as `results.tar.gz` in the current working directory.

```bash
python convert_tiff_to_float16.py
python check_and_prepare_data_for_upload.py ./results/
```
## Licensing

For licensing information, please see the `LICENSE` and `NOTICE` files.

Unless explicitly stated otherwise, the original code and documentation authored for this project are released under **CC BY-NC 4.0**. Third-party code, pretrained weights, datasets, and other external materials remain subject to their respective original licenses.

## Acknowledgements

This project builds upon ideas and code from [**INP-Former**](https://github.com/luow23/INP-Former), [**CPR**](https://github.com/flyinghu123/CPR), [**RoBiS**](https://github.com/xrli-U/RoBiS), [**DINOv3**](https://github.com/facebookresearch/dinov3), [**SAM 3**](https://github.com/facebookresearch/sam3), and [**HQ-SAM**](https://github.com/SysCV/SAM-HQ).

We sincerely thank the authors for making their work publicly available.

If you find this project helpful, please also consider citing their original work.