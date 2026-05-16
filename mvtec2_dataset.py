import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json
import re

torch.multiprocessing.set_sharing_strategy('file_system')

import math
import torch
from torchvision import transforms


class RandomLightingSimulation(object):
    def __init__(
        self,
        p=0.5,
        p_exposure=0.5,
        p_gradient=0.5,
        p_shadow=0.4,
        p_highlight=0.4, # 0.6
        p_vignette=0.3, # 0.4
        p_noise=0.5, # 0.2
        exposure_range=(-0.4, 0.4),
        gamma_range=(0.85, 1.2),
        gradient_strength=(0.08, 0.28),
        shadow_count=(1, 3),
        shadow_strength=(0.10, 0.35),
        shadow_sigma=(0.18, 0.45),
        highlight_count=(1, 2),
        highlight_strength=(0.04, 0.16),
        highlight_sigma=(0.08, 0.25),
        vignette_strength=(0.08, 0.25),
        noise_std_range=(0.0, 0.01),
    ):
        self.p = p
        self.p_exposure = p_exposure
        self.p_gradient = p_gradient
        self.p_shadow = p_shadow
        self.p_highlight = p_highlight
        self.p_vignette = p_vignette
        self.p_noise = p_noise

        self.exposure_range = exposure_range
        self.gamma_range = gamma_range
        self.gradient_strength = gradient_strength

        self.shadow_count = shadow_count
        self.shadow_strength = shadow_strength
        self.shadow_sigma = shadow_sigma

        self.highlight_count = highlight_count
        self.highlight_strength = highlight_strength
        self.highlight_sigma = highlight_sigma

        self.vignette_strength = vignette_strength
        self.noise_std_range = noise_std_range

    def _rand_uniform(self, low, high, device, dtype):
        return torch.empty(1, device=device, dtype=dtype).uniform_(low, high).item()

    def _make_coord_grid(self, h, w, device, dtype):
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        try:
            yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        except TypeError:
            yy, xx = torch.meshgrid(yy, xx)
        return yy, xx

    def _apply_exposure_and_gamma(self, img):
        device, dtype = img.device, img.dtype

        ev = self._rand_uniform(
            self.exposure_range[0],
            self.exposure_range[1],
            device,
            dtype,
        )
        img = img * (2.0 ** ev)
        img = torch.clamp(img, 0.0, 1.0)

        gamma = self._rand_uniform(
            self.gamma_range[0],
            self.gamma_range[1],
            device,
            dtype,
        )
        img = img ** gamma

        return img

    def _apply_directional_gradient(self, img, yy, xx):
        device, dtype = img.device, img.dtype

        angle = self._rand_uniform(0.0, 2.0 * math.pi, device, dtype)
        direction = math.cos(angle) * xx + math.sin(angle) * yy
        direction = direction / (direction.abs().max() + 1e-6)

        strength = self._rand_uniform(
            self.gradient_strength[0],
            self.gradient_strength[1],
            device,
            dtype,
        )
        if torch.rand(1, device=device).item() < 0.5:
            strength = -strength

        lighting = 1.0 + strength * direction
        lighting = torch.clamp(lighting, 0.55, 1.45)

        return img * lighting.unsqueeze(0)

    def _apply_shadow_blobs(self, img, yy, xx):
        device, dtype = img.device, img.dtype

        min_count, max_count = self.shadow_count
        count = torch.randint(min_count, max_count + 1, (1,), device=device).item()

        lighting = torch.ones_like(yy)

        for _ in range(count):
            cx = self._rand_uniform(-0.8, 0.8, device, dtype)
            cy = self._rand_uniform(-0.8, 0.8, device, dtype)
            sx = self._rand_uniform(
                self.shadow_sigma[0],
                self.shadow_sigma[1],
                device,
                dtype,
            )
            sy = self._rand_uniform(
                self.shadow_sigma[0],
                self.shadow_sigma[1],
                device,
                dtype,
            )
            angle = self._rand_uniform(0.0, math.pi, device, dtype)

            x_shift = xx - cx
            y_shift = yy - cy

            x_rot = math.cos(angle) * x_shift + math.sin(angle) * y_shift
            y_rot = -math.sin(angle) * x_shift + math.cos(angle) * y_shift

            blob = torch.exp(-0.5 * ((x_rot / sx) ** 2 + (y_rot / sy) ** 2))

            strength = self._rand_uniform(
                self.shadow_strength[0],
                self.shadow_strength[1],
                device,
                dtype,
            )
            lighting = lighting * (1.0 - strength * blob)

        lighting = torch.clamp(lighting, 0.45, 1.0)

        return img * lighting.unsqueeze(0)

    def _apply_highlight_blobs(self, img, yy, xx):
        device, dtype = img.device, img.dtype

        min_count, max_count = self.highlight_count
        count = torch.randint(min_count, max_count + 1, (1,), device=device).item()

        highlight = torch.zeros_like(yy)

        for _ in range(count):
            cx = self._rand_uniform(-0.8, 0.8, device, dtype)
            cy = self._rand_uniform(-0.8, 0.8, device, dtype)
            sx = self._rand_uniform(
                self.highlight_sigma[0],
                self.highlight_sigma[1],
                device,
                dtype,
            )
            sy = self._rand_uniform(
                self.highlight_sigma[0],
                self.highlight_sigma[1],
                device,
                dtype,
            )
            angle = self._rand_uniform(0.0, math.pi, device, dtype)

            x_shift = xx - cx
            y_shift = yy - cy

            x_rot = math.cos(angle) * x_shift + math.sin(angle) * y_shift
            y_rot = -math.sin(angle) * x_shift + math.cos(angle) * y_shift

            blob = torch.exp(-0.5 * ((x_rot / sx) ** 2 + (y_rot / sy) ** 2))

            strength = self._rand_uniform(
                self.highlight_strength[0],
                self.highlight_strength[1],
                device,
                dtype,
            )
            highlight = highlight + strength * blob

        return img + highlight.unsqueeze(0)

    def _apply_vignette(self, img, yy, xx):
        device, dtype = img.device, img.dtype

        radius = torch.sqrt(xx ** 2 + yy ** 2)
        radius = radius / (radius.max() + 1e-6)

        strength = self._rand_uniform(
            self.vignette_strength[0],
            self.vignette_strength[1],
            device,
            dtype,
        )
        lighting = 1.0 - strength * (radius ** 2)
        lighting = torch.clamp(lighting, 0.55, 1.0)

        return img * lighting.unsqueeze(0)

    def _apply_noise(self, img):
        device, dtype = img.device, img.dtype

        noise_std = self._rand_uniform(
            self.noise_std_range[0],
            self.noise_std_range[1],
            device,
            dtype,
        )
        if noise_std > 0:
            img = img + torch.randn_like(img) * noise_std

        return img

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        img = img.clone()
        _, h, w = img.shape
        device, dtype = img.device, img.dtype
        yy, xx = self._make_coord_grid(h, w, device, dtype)

        if torch.rand(1, device=device).item() < self.p_exposure:
            img = self._apply_exposure_and_gamma(img)

        if torch.rand(1, device=device).item() < self.p_gradient:
            img = self._apply_directional_gradient(img, yy, xx)

        if torch.rand(1, device=device).item() < self.p_shadow:
            img = self._apply_shadow_blobs(img, yy, xx)

        if torch.rand(1, device=device).item() < self.p_highlight:
            img = self._apply_highlight_blobs(img, yy, xx)

        if torch.rand(1, device=device).item() < self.p_vignette:
            img = self._apply_vignette(img, yy, xx)

        img = torch.clamp(img, 0.0, 1.0)

        if torch.rand(1, device=device).item() < self.p_noise:
            img = self._apply_noise(img)

        return torch.clamp(img, 0.0, 1.0)


def get_data_transforms(size, isize, mean_train=None, std_train=None, lighting_p=0.5):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train

    data_transforms_train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        RandomLightingSimulation(p=lighting_p),
        transforms.Normalize(mean=mean_train,
                             std=std_train)
                             ])
    
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)
                             ])
    
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size),interpolation=transforms.InterpolationMode.NEAREST),
        # transforms.Resize((size, size)),
        transforms.ToTensor()])

    return data_transforms, gt_transforms, data_transforms_train

class MVTec2Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase, resize, normal_only=True):
        self.phase = phase
        self.resize = resize
        self.transform = transform
        self.root = root
        self.normal_only = normal_only

        if phase == 'train':
            self.img_path_good = os.path.join(root, 'train', 'good')
            self.img_path_bad = os.path.join(root, 'train', 'bad')
            self.gt_path_bad = os.path.join(root, 'train', 'bad_mask')


        elif phase == 'test':
            self.img_path_private = os.path.join(root, 'test_private')
            self.img_path_private_mixed = os.path.join(root, 'test_private_mixed')
            self.gt_path = None

        elif phase == 'true_val':
            self.img_path_good = os.path.join(root, 'validation', 'good')

        else:
            raise ValueError(f"Unsupported phase: {phase}")

        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def generate_gt_filename(self,filename):
        name, ext = os.path.splitext(filename)
        # 如果已经有 _mask，直接返回原名
        if '_mask' in name:
            return name + ext

        # 先找分割的后缀
        # 支持 _grid4x2_0_0, _longedge4_3 等多种新后缀
        pattern = r'(_grid\d+x\d+_\d+_\d+|_longedge\d+_\d+)$'
        match = re.search(pattern, name)
        if match:
            idx = match.start()
            prefix = name[:idx]
            suffix = name[idx:]
            gt_name = prefix + '_mask' + suffix
        else:
            gt_name = name + '_mask'

        return gt_name + ext

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if self.phase == 'train':
            # 加载 good 图像
            good_paths = glob.glob(os.path.join(self.img_path_good, '*.png')) + \
                        glob.glob(os.path.join(self.img_path_good, '*.jpg')) + \
                        glob.glob(os.path.join(self.img_path_good, '*.bmp'))
            
            img_tot_paths.extend(good_paths)
            gt_tot_paths.extend([0] * len(good_paths))  # 没有 GT，为 0
            tot_labels.extend([0] * len(good_paths))
            tot_types.extend(['good'] * len(good_paths))

        
        elif self.phase == 'test':
            private_paths = glob.glob(os.path.join(self.img_path_private, '*.png')) + \
                            glob.glob(os.path.join(self.img_path_private, '*.jpg')) + \
                            glob.glob(os.path.join(self.img_path_private, '*.bmp'))
            private_mixed_paths = glob.glob(os.path.join(self.img_path_private_mixed, '*.png')) + \
                                  glob.glob(os.path.join(self.img_path_private_mixed, '*.jpg')) + \
                                  glob.glob(os.path.join(self.img_path_private_mixed, '*.bmp'))
            img_paths = private_paths + private_mixed_paths
            img_paths.sort()

            img_tot_paths.extend(img_paths)
            gt_tot_paths.extend([0] * len(img_paths))
            tot_labels.extend([0] * len(img_paths))
            tot_types.extend(['good'] * len(img_paths))
        
        elif self.phase == 'true_val':
            good_paths = glob.glob(os.path.join(self.img_path_good, '*.png')) + \
            glob.glob(os.path.join(self.img_path_good, '*.jpg')) + \
            glob.glob(os.path.join(self.img_path_good, '*.bmp'))
            
            img_tot_paths.extend(good_paths)
            gt_tot_paths.extend([0] * len(good_paths))  # 没有 GT，为 0
            tot_labels.extend([0] * len(good_paths))
            tot_types.extend(['good'] * len(good_paths))
        
        else:
            raise ValueError(f"Unsupported phase: {self.phase}")

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-1]])
        else:
            gt = Image.open(gt).convert('L')

        assert img.size()[1:] == gt.size()[1:], f"Image and GT size mismatch: {img.size()} vs {gt.size()}"

        return img, gt, label, img_path