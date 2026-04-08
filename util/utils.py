import numpy as np
import logging
import os
import time
import math

import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score
from copy import deepcopy


def upsample_size(x, size, mode='bilinear'):
    input_dtype = x.dtype

    if x.dim() == 3:
        x = x.unsqueeze(dim=1)
        keep_dim = True
    else:
        keep_dim = False

    x = x.float()
    if mode == 'bilinear':
        x = F.interpolate(x, size, mode=mode, align_corners=True)
    else:
        x = F.interpolate(x, size, mode=mode)

    if keep_dim:
        x = x.squeeze(dim=1)

    x = x.to(input_dtype)
    return x


def upsample_scale(x, scale, mode='bilinear'):
    input_dtype = x.dtype

    if x.dim() == 3:
        x = x.unsqueeze(dim=1)
        keep_dim = True
    else:
        keep_dim = False

    x = x.float()
    if mode == 'bilinear':
        x = F.interpolate(x, scale_factor=scale, mode=mode, align_corners=True)
        if keep_dim:
            x = x.squeeze(dim=1)
    else:
        x = F.interpolate(x, scale_factor=scale, mode=mode)
        if keep_dim:
            x = x.squeeze(dim=1)

    x = x.to(input_dtype)
    return x


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def generate_color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3
            cmap[i] = np.array([r, g, b])
        cmap[0] = np.array([255, 255, 255])
            
    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])
        cmap[0] = np.array([255, 255, 255])

    elif 'buildings' in dataset:
        cmap[0] = np.array([0, 0, 0])
        cmap[1] = np.array([255, 255, 255])

    return cmap


def apply_color_map(prediction, dataset='pascal'):
    """
    Apply color map to an HxW prediction map to convert it to an RGB color image.
    
    Parameters:
        prediction (numpy.ndarray): The prediction map with shape HxW.
        dataset (str): The dataset type, either 'pascal', 'coco', or 'cityscapes'.
        
    Returns:
        numpy.ndarray: The colorized prediction map with shape HxWx3.
    """
    # Get the color map for the specified dataset
    cmap = generate_color_map(dataset)
    
    # Prepare an output array with 3 channels for RGB, initializing to zero
    [H, W] = prediction.shape
    colorized_output = np.zeros((H, W, 3), dtype=np.uint8)

    # Apply color mapping
    colorized_output = cmap[prediction]
    
    return colorized_output


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def init_log(name, level=logging.INFO):
    logs = set()
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


class Evaluator(object):
    def __init__(self, num_class, ignore_index=None):
        self.num_class = num_class
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc_class = np.nanmean(Acc)  # nanmean is used to ignore NaN values which may arise if any class was ignored completely
        return Acc_class

    def Precision_Recall(self):
        Precision = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=0) + 1e-10)
        Recall = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + 1e-10)
        return Precision, Recall

    def F1_Score(self):
        Precision, Recall = self.Precision_Recall()
        F1 = 2 * Precision * Recall / (Precision + Recall)
        return F1

    def Mean_F1_Score(self):
        F1 = self.F1_Score()
        Mean_F1 = np.nanmean(F1)
        return Mean_F1

    def Mean_Intersection_over_Union(self):
        IoU = self.Intersection_over_Union()        
        valid_classes = np.arange(self.num_class) != self.ignore_index        
        mIoU = np.nanmean(IoU[valid_classes])
        return mIoU

    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        return IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Confusion_Matrix(self, normalize=False):
        """
        Return the normalized confusion matrix.

        Returns:
            np.array: The normalized confusion matrix.
        """
        # Avoid division by zero
        if normalize:
            with np.errstate(divide='ignore', invalid='ignore'):
                # Normalize the confusion matrix by row (i.e., by the number of samples in each actual class)
                normalized_confusion_matrix = self.confusion_matrix / self.confusion_matrix.sum(axis=1, keepdims=True)
                # Replace NaNs with 0 for classes that do not appear in the ground truth
                normalized_confusion_matrix = np.nan_to_num(normalized_confusion_matrix)
            return normalized_confusion_matrix
        else:
            return self.confusion_matrix

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class) & (gt_image != self.ignore_index)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        if isinstance(gt_image, torch.Tensor):
            gt_image = gt_image.cpu().numpy()
        if isinstance(pre_image, torch.Tensor):
            pre_image = pre_image.cpu().numpy()
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def obtain_cutmix(input_tensor, p=0.5, beta=1.0, min_cut_rat=0.1, max_cut_rat=0.75):
    B, C, H, W = input_tensor.shape
    device = input_tensor.device
    indices = torch.randperm(B).to(device)
    
    # Ensure everything is a tensor
    if torch.rand(1).item() < p:
        lamb = torch.distributions.Beta(beta, beta).sample([]).to(device)
        one = torch.tensor(1.0, device=device)  # Ensure constants are tensors
        min_cut_rat = torch.tensor(min_cut_rat, device=device)
        max_cut_rat = torch.tensor(max_cut_rat, device=device)

        # Use tensors for all operations
        cut_rat = (max_cut_rat - min_cut_rat) * torch.sqrt(one - lamb) + min_cut_rat
        cut_w = torch.round(W * cut_rat).type(torch.long)
        cut_h = torch.round(H * cut_rat).type(torch.long)

        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()

        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)
    else:
        bbx1 = bbx2 = bby1 = bby2 = 0  # Default coords for non-CutMix case

    coords = {'bbx1': bbx1, 'bby1': bby1, 'bbx2': bbx2, 'bby2': bby2}
    return coords, indices


def apply_cutmix(input_tensor, coords, indices):
    # Check the input dimensions, if 3D, assume the channel dimension is missing and add it
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(1)  # Add a single channel dimension
        was_three_dim = True  # Mark that the original input was 3D
    else:
        was_three_dim = False

    # Extract the bounding box coordinates from the coords dictionary
    bbx1, bby1, bbx2, bby2 = coords['bbx1'], coords['bby1'], coords['bbx2'], coords['bby2']
    
    # Perform the CutMix operation using the same coordinates and indices
    output = input_tensor.clone()
    output[:, :, bby1:bby2, bbx1:bbx2] = input_tensor[indices, :, bby1:bby2, bbx1:bbx2]
    
    # If the original input was 3D, remove the added channel dimension
    if was_three_dim:
        output = output.squeeze(1)
    return output