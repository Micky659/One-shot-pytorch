import torch
import re
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os


class Trainset(Dataset):

    def __init__(self, path):
        self.path = path
        self.transform = self.load_transforms()
        self.images = []

        for subFolder in os.listdir(self.path):
            for employee in os.listdir(os.path.join(self.path, subFolder)):
                for image in os.listdir(os.path.join(self.path, subFolder, employee)):
                    if re.search('script',
                                 os.path.basename(os.path.join(self.path, subFolder, employee, image))) is not None:
                        img1 = os.path.join(self.path, subFolder, employee, image)
                    else:
                        img2 = os.path.join(self.path, subFolder, employee, image)
                if img1 is not None and img2 is not None:
                    self.images.append([img1, img2])
                img1 = None
                img2 = None

    def __getitem__(self, item):
        imageSet = random.choice(self.images)

        passport = Image.open(imageSet[0])
        selfie = Image.open(imageSet[1])

        img1 = passport.convert("L")
        img2 = selfie.convert("L")

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, torch.from_numpy(np.array([int(passport != selfie)], dtype=np.float32))

    def load_transforms(self):
        tet = transforms.Compose([transforms.Resize((100, 100)),
                                  transforms.RandomAffine(15),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ColorJitter(
                                      brightness=0.3,
                                      contrast=0.3,
                                      saturation=0.3),
                                  transforms.ToTensor()])
        return tet

    def __len__(self):
        images = []
        for subFolder in os.listdir(self.path):
            for employee in os.listdir(os.path.join(self.path, subFolder)):
                for image in os.listdir(os.path.join(self.path, subFolder, employee)):
                    images.append(image)
        return len(images)

