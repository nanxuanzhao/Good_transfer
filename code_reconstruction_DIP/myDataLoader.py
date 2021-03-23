import torch.utils.data
import torch.utils.data as data
import os
import os.path
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import numpy as np
from numpy import random
import glob
from matplotlib import pyplot as plt

class myDataLoader(data.Dataset):

    def __init__(self, root, initial_size=256, img_size=224):
        super(myDataLoader, self).__init__()
        self.root = root
        self.initial_size = initial_size
        self.img_size = img_size

        classes, class_to_idx = self._find_classes(self.root)
        img_list = glob.glob(os.path.join(root,'*','*.JPEG'))
        if len(img_list) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)
        print("Found {} files in subfolders of: {}\n".format(len(img_list),self.root))

        img_list_per_class = {}
        for temp_class in classes:
            img_list_per_class[temp_class] = glob.glob(os.path.join(self.root,temp_class,'*.JPEG'))

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.img_list = img_list
        self.img_list_per_class = img_list_per_class

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
                            transforms.Resize(self.initial_size),
                            transforms.CenterCrop(self.img_size),
                            transforms.ToTensor(),
                            normalize,
                        ])

        self.detransform = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.ToPILImage()
            ])

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        """
        path = self.img_list[index]


        #same sample with different transform
        img = self._pil_loader(path)
        sample = self.transform(img)

        temp_class = path.split('/')[-2]
        target = self.class_to_idx[temp_class]
        filename = os.path.join(temp_class,path.split('/')[-1])

        return sample, target, filename

    def __len__(self):
        return len(self.img_list)