import random
import numpy as np
import colorsys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.transforms import ColorJitter, RandomGrayscale
from PIL import Image, ImageOps, ImageFilter


def crop(rgb, dpt, mask, size, pad_value=0):
    w, h = rgb.size
    pad_w = size - w if w < size else 0
    pad_h = size - h if h < size else 0
    rgb = ImageOps.expand(rgb, border=(0, 0, pad_w, pad_h), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=pad_value)
    dpt = ImageOps.expand(dpt, border=(0, 0, pad_w, pad_h), fill=dpt.getextrema()[0][0])

    w, h = rgb.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    rgb = rgb.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))
    dpt = dpt.crop((x, y, x + size, y + size))
    return rgb, dpt, mask


def hflip(rgb, dpt, mask, p=0.5):
    if random.random() < p:
        rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        dpt = dpt.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return rgb, dpt, mask


def resize(rgb, dpt, mask, ratio_range):
    w, h = rgb.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    rgb = rgb.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    dpt = dpt.resize((ow, oh), Image.NEAREST)
    return rgb, dpt, mask


def blur(rgb, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=sigma))
    return rgb


def normalize_rgb(rgb):
    rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(rgb)
    return rgb


def normalize_dpt(dpt):
    dpt = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
    ])(dpt)
    return dpt


