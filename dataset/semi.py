from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, mode='val', class_bd=0, class_bg=255, \
                    size=None, id_path=None, nsample=None):
        self.name = name
        self.mode = mode
        self.size = size
        self.class_bd = class_bd
        self.class_bg = class_bg

        self.root = root
        if '/dss/dssmcmlfs01/pn36te/pn36te-dss-0000/ge89dol2' in os.path.abspath(__file__):
            self.root = root.replace('home', '/dss/dssmcmlfs01/pn36te/pn36te-dss-0000/ge89dol2')

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]

        splits = id.split(' ')
        rgb_id, mask_id, dpt_id = splits[0], splits[1], splits[2]

        rgb_path = os.path.join(self.root, rgb_id)
        dpt_path = os.path.join(self.root, dpt_id)
        mask_path = os.path.join(self.root, mask_id)

        rgb = Image.open(rgb_path).convert('RGB')
        dpt = Image.open(dpt_path).convert('RGB')

        mask = Image.open(mask_path).convert('L')
        mask = Image.fromarray((np.array(mask) == self.class_bd).astype(np.uint8))

        if self.mode == 'val':
            rgb = normalize_rgb(rgb)
            dpt = normalize_dpt(dpt)
            mask = torch.from_numpy(np.array(mask)).long()
            return rgb, dpt, mask, id

        elif self.mode == 'train_l':
            rgb, dpt, mask = resize(rgb, dpt, mask, (0.5, 2.0))
            rgb, dpt, mask = crop(rgb, dpt, mask, self.size, self.class_bg)
            rgb, dpt, mask = hflip(rgb, dpt, mask, p=0.5)

            rgb = normalize_rgb(rgb)
            dpt = normalize_dpt(dpt)
            mask = torch.from_numpy(np.array(mask)).long()
            return rgb, dpt, mask

        elif self.mode == 'train_u':
            rgb, dpt, mask = resize(rgb, dpt, mask, (0.5, 2.0))
            rgb, dpt, mask = crop(rgb, dpt, mask, self.size, self.class_bg)
            rgb, dpt, mask = hflip(rgb, dpt, mask, p=0.5)

            rgb_w, rgb_s = deepcopy(rgb), deepcopy(rgb)
            dpt_w, dpt_s = deepcopy(dpt), deepcopy(dpt)

            if random.random() < 0.8:
                rgb_s = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)(rgb_s)
                dpt_s = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)(dpt_s)
            rgb_s = transforms.RandomGrayscale(p=0.2)(rgb_s)
            dpt_s = transforms.RandomGrayscale(p=0.2)(dpt_s)
            rgb_s = blur(rgb_s, p=0.5)
            dpt_s = blur(dpt_s, p=0.5)

            rgb_w = normalize_rgb(rgb_w)
            dpt_w = normalize_dpt(dpt_w)
            rgb_s = normalize_rgb(rgb_s)
            dpt_s = normalize_dpt(dpt_s)
            mask = torch.from_numpy(np.array(mask)).long()
            return rgb_w, rgb_s, dpt_w, dpt_s, mask, id

    def __len__(self):
        return len(self.ids)



class SemiDataset_UniMatch(Dataset):
    def __init__(self, name, root, mode='val', class_bd=0, class_bg=255, \
                    size=None, id_path=None, nsample=None):
        self.name = name
        self.mode = mode
        self.size = size
        self.class_bd = class_bd
        self.class_bg = class_bg

        self.root = root
        if '/dss/dssmcmlfs01/pn36te/pn36te-dss-0000/ge89dol2' in os.path.abspath(__file__):
            self.root = root.replace('home', '/dss/dssmcmlfs01/pn36te/pn36te-dss-0000/ge89dol2')

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]

        splits = id.split(' ')
        rgb_id, mask_id, dpt_id = splits[0], splits[1], splits[2]

        rgb_path = os.path.join(self.root, rgb_id)
        dpt_path = os.path.join(self.root, dpt_id)
        mask_path = os.path.join(self.root, mask_id)

        rgb = Image.open(rgb_path).convert('RGB')
        dpt = Image.open(dpt_path).convert('RGB')

        mask = Image.open(mask_path).convert('L')
        mask = Image.fromarray((np.array(mask) == self.class_bd).astype(np.uint8))

        if self.mode == 'val':
            rgb = normalize_rgb(rgb)
            dpt = normalize_dpt(dpt)
            mask = torch.from_numpy(np.array(mask)).long()
            return rgb, mask, id

        elif self.mode == 'train_l':
            rgb, dpt, mask = resize(rgb, dpt, mask, (0.5, 2.0))
            rgb, dpt, mask = crop(rgb, dpt, mask, self.size, self.class_bg)
            rgb, dpt, mask = hflip(rgb, dpt, mask, p=0.5)

            rgb = normalize_rgb(rgb)
            dpt = normalize_dpt(dpt)
            mask = torch.from_numpy(np.array(mask)).long()
            return rgb, mask

        elif self.mode == 'train_u':
            rgb, dpt, mask = resize(rgb, dpt, mask, (0.5, 2.0))
            rgb, dpt, mask = crop(rgb, dpt, mask, self.size, self.class_bg)
            rgb, dpt, mask = hflip(rgb, dpt, mask, p=0.5)

            rgb_w, rgb_s1, rgb_s2 = deepcopy(rgb), deepcopy(rgb), deepcopy(rgb)

            if random.random() < 0.8:
                rgb_s1 = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)(rgb_s1)
                rgb_s2 = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)(rgb_s2)
            rgb_s1 = transforms.RandomGrayscale(p=0.2)(rgb_s1)
            rgb_s2 = transforms.RandomGrayscale(p=0.2)(rgb_s2)
            rgb_s1 = blur(rgb_s1, p=0.5)
            rgb_s2 = blur(rgb_s2, p=0.5)

            rgb_w = normalize_rgb(rgb_w)
            rgb_s1 = normalize_rgb(rgb_s1)
            rgb_s2 = normalize_rgb(rgb_s2)
            mask = torch.from_numpy(np.array(mask)).long()
            return rgb_w, rgb_s1, rgb_s2

    def __len__(self):
        return len(self.ids)