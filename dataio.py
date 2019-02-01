import torch
import torchvision
from torchvision.transforms import *


class NoisyCIFAR10Dataset():
    '''Dataset class that adds noise to cifar10-images'''
    def __init__(self,
                 data_root,
                 sigma,
                 train):
        super().__init__()

        self.sigma = sigma

        self.transforms = Compose([
            ToTensor()
        ])

        self.cifar10 = torchvision.datasets.CIFAR10(root=data_root,
                                                    train=train,
                                                    download=True,
                                                    transform=self.transforms)

    def __len__(self):
        return len(self.cifar10)

    def add_noise(self, img):
        return img + torch.randn_like(img) * self.sigma

    def __getitem__(self, idx):
        '''Returns tuple of (model_input, ground_truth)'''
        img, _ = self.cifar10[idx]

        img = (img - 0.5) * 2
        noisy_img = self.add_noise(img)

        return noisy_img, img
