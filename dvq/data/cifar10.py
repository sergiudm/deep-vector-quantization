from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

import lightning.pytorch as pl


class CIFAR10Data(pl.LightningDataModule):
    """returns cifar-10 examples in floats in range [0,1]"""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4, padding_mode="reflect"),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
        dataset = CIFAR10(
            root=self.hparams.data_dir, train=True, transform=transform, download=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=False,
            shuffle=True,
            persistent_workers=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        dataset = CIFAR10(
            root=self.hparams.data_dir, train=False, transform=transform, download=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=False,
            persistent_workers=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
