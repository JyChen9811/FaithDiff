import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from functools import partial
import torch.nn.functional as F
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from basicsr.utils.img_process_util import filter2D
from PIL import Image
import json
from transformers import CLIPImageProcessor
from torch import nn
from torchvision import transforms
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from .realesrgan import RealESRGAN_degradation
import cv2
import random
from glob import glob
from collections import OrderedDict
import yaml
from PIL import Image
def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def opt_parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)  # ignore_security_alert_wait_for_fix RCE

    return opt

def convert_image_to_fn(img_type, image, minsize=512, eps=0.02):
    width, height = image.size
    if min(width, height) < minsize:
        scale = minsize/min(width, height) + eps
        image = image.resize((math.ceil(width*scale), math.ceil(height*scale)))

    if image.mode != img_type:
        return image.convert(img_type)
    return image
def exists(x):
    return x is not None


class LocalImageDataset(data.Dataset):
    def __init__(self, 
                img_file = None,
                face_file = None,
                yml_kernel = None,
                image_size=512,
                tokenizer=None,
                tokenizer_2=None,
                center_crop=False,
                random_flip=True,
                resize_bak=True,
                convert_image_to="RGB",
                t_drop_rate=0.05
        ):
        super(LocalImageDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2

        self.resize_bak = resize_bak

        self.crop_size = image_size


        self.t_drop_rate = t_drop_rate



        nature_paths = []
        nature_lr_paths = []
        nature_jsons = []

        face_paths = []
        lq_face_paths = []
        face_jsons = []



        self.data_types = ['nature', 'face']
        self.data_prob = [0.875, 0.125]
        for img_path_idx in img_file[0]:
            img_path_list = sorted(glob(os.path.join(img_path_idx, '**', '*.png'), recursive=True))
            nature_paths += img_path_list

        for lq_img_path_idx in img_file[1]:
            lq_img_path_list = sorted(glob(os.path.join(lq_img_path_idx, '**', '*.png'), recursive=True))
            nature_lr_paths += lq_img_path_list

        for text_path_idx in img_file[2]:
            text_path_list = sorted(glob(os.path.join(text_path_idx, '**', '*.json'), recursive=True))
            nature_jsons += text_path_list

        for face_path_idx in face_file[0]:
            face_path_list = sorted(glob(os.path.join(face_path_idx, '**', '*.png'), recursive=True))
            face_paths += face_path_list

        for lq_face_path_idx in face_file[1]:
            lq_face_path_list = sorted(glob(os.path.join(lq_face_path_idx, '**', '*.png'), recursive=True))
            lq_face_paths += lq_face_path_list

        for face_text_path_idx in face_file[2]:
            face_text_path_list = sorted(glob(os.path.join(face_text_path_idx, '**', '*.json'), recursive=True))
            face_jsons += face_text_path_list


        self.data_collection = {'nature': (np.array(nature_paths), np.array(nature_jsons), np.array(nature_lr_paths)), 'face': (np.array(face_paths), np.array(face_jsons), np.array(lq_face_paths))}
        self.data_lens = {'nature': len(nature_paths), 'face': len(face_paths)}
        print(self.data_lens)
        self.data_lens = {'nature': len(nature_jsons),  'face': len(face_jsons)}
        print(self.data_lens)
        self.data_lens = {'nature': len(nature_lr_paths), 'face': len(lq_face_paths)}
        print(self.data_lens)

        self.datatypes_lens = [len(nature_paths), len(face_paths)]
        self.cumulative_lens = np.cumsum([0] + self.datatypes_lens)
    def __getitem__(self, index):

        data_type_idx = np.where(self.cumulative_lens <= index )[0][-1]

        data_type = self.data_types[data_type_idx]
        index = index - self.cumulative_lens[data_type_idx]

        crop_pad_size = self.crop_size
        # load image
        img_path = self.data_collection[data_type][0][index]
        json_path = self.data_collection[data_type][1][index]
        lq_img_path = self.data_collection[data_type][2][index]
        gt_path = img_path
        data = json.load(open(json_path)) 
        init_text = data["caption"]
        words = init_text.split()
        words = words[3:]
        words[0] = words[0].capitalize()
        text = ' '.join(words)
        text = text.split('. ')
        text = '. '.join(text[:2]) + '.'

        image = Image.open(img_path).convert('RGB')
        
        if 'FFHQ' in lq_img_path:
            if random.random() < 0.5:
                lq_img_path = lq_img_path.replace('LR_crops_1', 'LR_crops_2')
        

        lq_image = Image.open(lq_img_path).convert('RGB')

        if 'FFHQ' in img_path:
            random_size = random.randint(128, 192)
            lq_image = lq_image.resize((random_size, random_size), Image.BICUBIC)  
            image = image.resize((int(random_size * 4), int(random_size * 4)), Image.BICUBIC)  

        w, h = lq_image.size
        pil_img = np.array(image)  
        pil_lr_img = np.array(lq_image)   
        pil_img, pil_lr_img = augment([pil_img, pil_lr_img], hflip=True, rotation=False)

        crop_pad_size = self.crop_size // 4 
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            pil_lr_img = cv2.copyMakeBorder(pil_lr_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

            pad_h = max(0, self.crop_size - h * 4)
            pad_w = max(0, self.crop_size - w * 4)
            pil_img = cv2.copyMakeBorder(pil_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)


        # crop
        if pil_lr_img.shape[0] > crop_pad_size or pil_lr_img.shape[1] > crop_pad_size:
            h, w = pil_lr_img.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            
            pil_lr_img = pil_lr_img[top : top + crop_pad_size, left : left + crop_pad_size, ...]
            pil_img = pil_img[top * 4 : (top + crop_pad_size) * 4, left * 4: (left + crop_pad_size) * 4, ...]

        else:
            top = 0
            left = 0
        
        lq_image = Image.fromarray(pil_lr_img)
        mode = random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC])
        lr_w, lr_h = lq_image.size
        lq_image = lq_image.resize((lr_w * 4, lr_h * 4), mode)  
        
        image = Image.fromarray(pil_img)
        original_size = torch.tensor([h * 4, w * 4])
        crop_coords_top_left = torch.tensor([top * 4, left * 4])

        GT_image_t = np.asarray(image)/255.
        LR_image_t = np.asarray(lq_image)/255.

        GT_image_t, LR_image_t = img2tensor([GT_image_t, LR_image_t], bgr2rgb=False, float32=True)
        LR_image_t = LR_image_t * 2.0 - 1.0
        GT_image_t =  GT_image_t * 2.0 - 1.0

        rand_num = random.random()
        if rand_num < self.t_drop_rate:
            text = ""

        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        null_text_input_ids = self.tokenizer(
            '',
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        null_text_input_ids_2 = self.tokenizer_2(
            '',
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return {
            'lq_image': LR_image_t,
            "image": GT_image_t,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.crop_size, self.crop_size]),
            'gt_path': gt_path,
            'null_text_input_ids': null_text_input_ids,
            "null_text_input_ids_2": null_text_input_ids_2
            # "check_img": check_image
        }

    def __len__(self):
        total_length = 0
        for key, value in self.data_lens.items():
            total_length += value
        return total_length
