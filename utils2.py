import math
import os
import random
import logging
from functools import partial
from statistics import mean

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from numpy import ndarray
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from skimage import measure
from torch.nn import functional as F
from tqdm import tqdm

from adeval import EvalAccumulatorCuda
from aug_funcs import grey_img, hflip_img, rot90_img, rot_img, translation_img


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def ader_evaluator(pr_px, pr_sp, gt_px, gt_sp, use_metrics=['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO']):
    if len(gt_px.shape) == 4:
        gt_px = gt_px.squeeze(1)
    if len(pr_px.shape) == 4:
        pr_px = pr_px.squeeze(1)

    score_min = min(pr_sp)
    score_max = max(pr_sp)
    anomap_min = pr_px.min()
    anomap_max = pr_px.max()

    accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max, skip_pixel_aupro=False, nstrips=200)
    accum.add_anomap_batch(torch.tensor(pr_px).cuda(non_blocking=True), torch.tensor(gt_px.astype(np.uint8)).cuda(non_blocking=True))

    metrics = accum.summary()
    metric_results = {}
    for metric in use_metrics:
        if metric.startswith('I-AUROC'):
            metric_results[metric] = roc_auc_score(gt_sp, pr_sp)
        elif metric.startswith('I-AP'):
            metric_results[metric] = average_precision_score(gt_sp, pr_sp)
        elif metric.startswith('I-F1_max'):
            metric_results[metric] = f1_score_max(gt_sp, pr_sp)
        elif metric.startswith('P-AUROC'):
            metric_results[metric] = metrics['p_auroc']
        elif metric.startswith('P-AP'):
            metric_results[metric] = metrics['p_aupr']
        elif metric.startswith('P-F1_max'):
            metric_results[metric] = f1_score_max(gt_px.ravel(), pr_px.ravel())
        elif metric.startswith('AUPRO'):
            metric_results[metric] = metrics['p_aupro']
    return list(metric_results.values())


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    if logger.handlers:
        logger.handlers.clear()

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def augmentation(img):
    img = img.unsqueeze(0)
    augment_img = img
    for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, 3 * np.pi / 16, np.pi / 4]:
        rotate_img = rot_img(img, angle)
        augment_img = torch.cat([augment_img, rotate_img], dim=0)
    for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)]:
        trans_img = translation_img(img, a, b)
        augment_img = torch.cat([augment_img, trans_img], dim=0)
    augment_img = torch.cat([augment_img, hflip_img(img)], dim=0)
    augment_img = torch.cat([augment_img, grey_img(img)], dim=0)
    for angle in [1, 2, 3]:
        augment_img = torch.cat([augment_img, rot90_img(img, angle)], dim=0)
    return augment_img[torch.randperm(augment_img.size(0))]


def apply_gradient_weight(x, factor):
    factor = factor.expand_as(x).detach()
    return x * factor + x.detach() * (1 - factor)


def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x

def global_cosine_hm_adaptive(a, b, y=3):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()
        mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        # thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        factor = (point_dist/mean_dist)**(y)
        # factor = factor/torch.max(factor)
        # factor = torch.clip(factor, min=min_grad)
        # print(thresh)
        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))
        partial_func = partial(modify_grad_v2, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for fs, ft in zip(fs_list, ft_list):
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


def compose_anomaly_map(recon_map, seg_logits=None, seg_weight=0.35):
    if seg_logits is None:
        return recon_map
    seg_map = torch.sigmoid(seg_logits)
    if seg_map.shape[-2:] != recon_map.shape[-2:]:
        seg_map = F.interpolate(seg_map, size=recon_map.shape[-2:], mode='bilinear', align_corners=True)
    return recon_map + seg_weight * seg_map


def dice_bce_loss(logits, target, smooth=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + smooth) / (union + smooth))
    return bce + dice.mean()


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min + 1e-8)


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    return thrs[np.argmax(f1s)]


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    return f1s[:-1].max()


def denormalize(img):
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    return (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)


def restore_from_normalized_torch(x):
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return (x * std + mean).clamp(0.0, 1.0)


def normalize_to_imagenet(x):
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return (x - mean) / std


class DTDTextureBank:
    def __init__(self, dtd_root, resize=512, max_textures=2048):
        self.resize = int(resize)
        root = os.path.expanduser(dtd_root)
        candidates = []
        for base in [root, os.path.join(root, 'images')]:
            if os.path.isdir(base):
                for dirpath, _, filenames in os.walk(base):
                    for fname in filenames:
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            candidates.append(os.path.join(dirpath, fname))
        candidates = sorted(set(candidates))
        if len(candidates) == 0:
            raise FileNotFoundError(f'No DTD textures found under: {dtd_root}')
        if max_textures is not None and len(candidates) > max_textures:
            candidates = candidates[:max_textures]

        self.textures = []
        for path in candidates:
            with Image.open(path).convert('RGB') as img:
                img = img.resize((self.resize, self.resize), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.uint8)
            tex = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            self.textures.append(tex)

    def __len__(self):
        return len(self.textures)

    def sample(self, batch_size, image_size, device, dtype=torch.float32):
        h, w = image_size
        idxs = random.choices(range(len(self.textures)), k=batch_size)
        textures = torch.stack([self.textures[idx] for idx in idxs], dim=0)

        for i in range(batch_size):
            if random.random() < 0.5:
                textures[i] = torch.flip(textures[i], dims=[2])
            if random.random() < 0.5:
                textures[i] = torch.flip(textures[i], dims=[1])
            k = random.randint(0, 3)
            textures[i] = torch.rot90(textures[i], k, dims=[1, 2])

        textures = textures.to(device=device, dtype=dtype) / 255.0
        if textures.shape[-2:] != (h, w):
            textures = F.interpolate(textures, size=(h, w), mode='bilinear', align_corners=True)
        return textures


def _fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def generate_perlin_noise_2d(shape, res):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    g00 = gradients[:-1, :-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, :-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    n00 = np.sum(grid * g00[:shape[0], :shape[1]], 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10[:shape[0], :shape[1]], 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01[:shape[0], :shape[1]], 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11[:shape[0], :shape[1]], 2)

    t = _fade(grid[:shape[0], :shape[1]])
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def make_perlin_mask(height, width, scale_min=0, scale_max=6, threshold=0.5):
    scale_x = 2 ** random.randint(scale_min, scale_max)
    scale_y = 2 ** random.randint(scale_min, scale_max)
    noise = generate_perlin_noise_2d((height, width), (max(1, scale_x), max(1, scale_y)))
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    mask = (noise > threshold).astype(np.float32)
    if mask.sum() < 16:
        threshold = max(0.3, threshold - 0.15)
        mask = (noise > threshold).astype(np.float32)
    return mask


def synthesize_dtd_perlin(images, texture_bank, beta_min=0.35, beta_max=0.80, no_anomaly_prob=0.10, scale_min=0, scale_max=6):
    clean = restore_from_normalized_torch(images)
    b, _, h, w = clean.shape
    textures = texture_bank.sample(b, (h, w), device=clean.device, dtype=clean.dtype)
    synth = clean.clone()
    masks = torch.zeros((b, 1, h, w), device=clean.device, dtype=clean.dtype)

    for idx in range(b):
        if random.random() < no_anomaly_prob:
            continue
        threshold = random.uniform(0.45, 0.70)
        mask_np = make_perlin_mask(h, w, scale_min=scale_min, scale_max=scale_max, threshold=threshold)
        mask = torch.from_numpy(mask_np).to(device=clean.device, dtype=clean.dtype).unsqueeze(0)
        beta = random.uniform(beta_min, beta_max)
        blended = clean[idx] * (1.0 - mask) + ((1.0 - beta) * clean[idx] + beta * textures[idx]) * mask
        synth[idx] = blended.clamp(0.0, 1.0)
        masks[idx] = mask
    return normalize_to_imagenet(synth), masks


def save_imag_ZS(imgs, anomaly_map, gt, prototype_map, save_root, img_path):
    batch_num = imgs.shape[0]
    for i in range(batch_num):
        img_path_list = img_path[i].split('\\')
        class_name, category, idx_name = img_path_list[-4], img_path_list[-2], img_path_list[-1]
        os.makedirs(os.path.join(save_root, class_name, category), exist_ok=True)
        input_frame = denormalize(imgs[i].clone().squeeze(0).cpu().detach().numpy())
        cv2_input = np.array(input_frame, dtype=np.uint8)
        plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_0.png'), cv2_input)
        ano_map = anomaly_map[i].squeeze(0).cpu().detach().numpy()
        plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_1.png'), ano_map, cmap='jet')
        gt_map = gt[i].squeeze(0).cpu().detach().numpy()
        plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_2.png'), gt_map, cmap='gray')
        distance = prototype_map[i].view((28, 28)).cpu().detach().numpy()
        distance = cv2.resize(distance, (392, 392), interpolation=cv2.INTER_AREA)
        plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_3.png'), distance, cmap='jet')
        plt.close()


def evaluation_batch(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None, save_dir=None, dataset_root=None, normalize_amap=False):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    anomaly_map_list = []
    gt_mask_list = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, img_path_batch in tqdm(dataloader, ncols=80):
            img = img.to(device)
            target_size = resize_mask if resize_mask is not None else img.shape[-1]
            en, de, _, seg_logits = model(img, return_seg=True, seg_out_size=(target_size, target_size) if not isinstance(target_size, tuple) else target_size)
            anomaly_map_batch, _ = cal_anomaly_maps(en, de, target_size)
            anomaly_map_batch = compose_anomaly_map(anomaly_map_batch, seg_logits, getattr(model, 'eval_seg_weight', 0.0))

            if resize_mask is not None:
                anomaly_map_batch = F.interpolate(anomaly_map_batch, size=resize_mask, mode='bilinear', align_corners=True)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
                img = F.interpolate(img, size=resize_mask, mode='bilinear', align_corners=True)

            anomaly_map_batch = gaussian_kernel(anomaly_map_batch)

            if normalize_amap:
                scores = np.array([anomaly_map_batch[idx, 0].detach().cpu().numpy() for idx in range(anomaly_map_batch.shape[0])])
                min_scores = np.min(scores)
                max_scores = np.max(scores)
                anomaly_scores = (scores - min_scores) / (max_scores - min_scores + 1e-10)
                anomaly_scores = np.clip(anomaly_scores, 0, 1)
            else:
                anomaly_scores = []
                for idx in range(anomaly_map_batch.shape[0]):
                    anomaly_map = anomaly_map_batch[idx, 0].detach().cpu().numpy()
                    anomaly_scores.append(np.clip(anomaly_map, 0, 1))
                anomaly_scores = np.array(anomaly_scores)

            for idx, img_path in enumerate(img_path_batch):
                anomaly_map = anomaly_scores[idx]
                gt_mask = gt[idx, 0].detach().cpu().numpy()
                gt_mask = (gt_mask * 255).astype(np.uint8)
                gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
                anomaly_map_gray = (anomaly_map * 255).astype(np.uint8)
                anomaly_map_list.append(anomaly_map_gray)
                gt_mask_list.append(gt_mask[:, :, 0])

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(torch.tensor(anomaly_scores).unsqueeze(1).to(device))
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(torch.tensor(anomaly_scores).flatten(1).to(device), dim=1)[0]
            else:
                anomaly_map_flat = torch.tensor(anomaly_scores).flatten(1).to(device)
                sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][:, :int(anomaly_map_flat.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, anomaly_map_list, gt_mask_list]


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    assert isinstance(amaps, ndarray)
    assert isinstance(masks, ndarray)
    assert amaps.ndim == 3
    assert masks.ndim == 3
    assert amaps.shape == masks.shape
    assert set(masks.flatten()) == {0, 1}
    assert isinstance(num_th, int)

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        df = df._append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()
    return auc(df["fpr"], df["pro"])


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size // 2)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


from torch.optim.lr_scheduler import _LRScheduler


class WarmCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))
        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for _ in self.base_lrs]
        return [self.schedule[self.last_epoch] for _ in self.base_lrs]
