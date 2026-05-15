import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from monai import transforms
from monai.data import Dataset, DataLoader, pad_list_data_collate
from monai.utils import set_determinism
from monai.networks.nets import resnet10

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DualTransform:
    def __init__(self, keys):
        aug = [
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys),
            transforms.Orientationd(keys=keys, axcodes="RAS"),
            transforms.Spacingd(keys=keys, pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
            transforms.NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
            transforms.ResizeWithPadOrCropd(keys=keys, spatial_size=(128, 128, 128)),
            transforms.RandSpatialCropd(keys=keys, roi_size=(128, 128, 128), random_center=True),
            transforms.RandGaussianNoised(keys=keys, prob=0.2),
            transforms.ToTensord(keys=keys)
        ]
        self.view1 = transforms.Compose(aug)
        self.view2 = transforms.Compose(aug)

    def __call__(self, data):
        return {
            "view1": self.view1(data),
            "view2": self.view2(data),
        }


def dataset():
    keys = ['t1c', 't1n', 't2f', 't2w']
    transform = DualTransform(keys)

    def generate_splits(data_path):
        subjects = []
        for i in os.listdir(data_path):
            if not os.path.isdir(os.path.join(data_path, i)):
                continue
            if 'GLI' in i or 'UPENN' in i or 'UCSF' in i or 'Patient' in i:
                group = 0
            elif 'MET' in i:
                group = 1
            elif 'PED' in i:
                group = 2
            elif 'MEN' in i:
                group = 3
            else:
                group = -1

            subject = {k: os.path.join(data_path, i, f"{i}-{k}.nii.gz") for k in keys}
            subject['group'] = group
            subjects.append(subject)
        return subjects

    train_data = generate_splits('/path/to/train/')
    val_data = generate_splits('/path/to/val/')

    train_ds = Dataset(train_data, transform=transform)
    val_ds = Dataset(val_data, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=4,
                              collate_fn=pad_list_data_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=12, shuffle=False, num_workers=4,
                            collate_fn=pad_list_data_collate, drop_last=True)
    return train_loader, val_loader


class GroupDROLoss(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups
        self.register_buffer("group_losses", torch.zeros(num_groups))

    def forward(self, losses, group_ids):
        group_ids = torch.as_tensor(group_ids, device=losses.device)
        group_losses = []
        for g in range(self.num_groups):
            mask = (group_ids == g)
            if mask.any():
                group_loss = losses[mask].mean()
                group_losses.append(group_loss)
            else:
                group_losses.append(torch.tensor(0.0, device=losses.device))
        group_losses = torch.stack(group_losses)
        return group_losses.max()


class MLPHead(nn.Module):
    def __init__(self, in_dim=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class BYOLModule(pl.LightningModule):
    def __init__(self, lr=1e-4, beta=0.99):
        super().__init__()
        self.save_hyperparameters()

        self.online_encoder = resnet10(spatial_dims=3, n_input_channels=4, feed_forward=False)
        self.online_projector = MLPHead(in_dim=512, out_dim=128)
        self.online_predictor = MLPHead(in_dim=128, out_dim=128)

        self.target_encoder = resnet10(spatial_dims=3, n_input_channels=4, feed_forward=False)
        self.target_projector = MLPHead(in_dim=512, out_dim=128)

        self.group_dro = GroupDROLoss(num_groups=4)

        # Initialize target weights
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.hparams.beta + param_q.data * (1. - self.hparams.beta)
        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * self.hparams.beta + param_q.data * (1. - self.hparams.beta)

    def forward(self, x):
        return self.online_predictor(self.online_projector(self.online_encoder(x)))

    def byol_loss_per_sample(self, p, z):
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return 2 - 2 * (p * z).sum(dim=1)

    def training_step(self, batch, batch_idx):
        x1 = torch.cat([batch["view1"]["t1c"], batch["view1"]["t1n"],
                        batch["view1"]["t2f"], batch["view1"]["t2w"]], dim=1)
        x2 = torch.cat([batch["view2"]["t1c"], batch["view2"]["t1n"],
                        batch["view2"]["t2f"], batch["view2"]["t2w"]], dim=1)

        q1 = self.online_predictor(self.online_projector(self.online_encoder(x1)))
        q2 = self.online_predictor(self.online_projector(self.online_encoder(x2)))

        with torch.no_grad():
            self.momentum_update()
            z1 = self.target_projector(self.target_encoder(x1))
            z2 = self.target_projector(self.target_encoder(x2))

        loss1 = self.byol_loss_per_sample(q1, z2)
        loss2 = self.byol_loss_per_sample(q2, z1)
        all_losses = (loss1 + loss2) / 2
        group_ids = batch["view1"]["group"]
        loss = self.group_dro(all_losses, group_ids)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x1 = torch.cat([batch["view1"]["t1c"], batch["view1"]["t1n"],
                        batch["view1"]["t2f"], batch["view1"]["t2w"]], dim=1)
        x2 = torch.cat([batch["view2"]["t1c"], batch["view2"]["t1n"],
                        batch["view2"]["t2f"], batch["view2"]["t2w"]], dim=1)

        q1 = self.online_predictor(self.online_projector(self.online_encoder(x1)))
        q2 = self.online_predictor(self.online_projector(self.online_encoder(x2)))

        with torch.no_grad():
            z1 = self.target_projector(self.target_encoder(x1))
            z2 = self.target_projector(self.target_encoder(x2))

        loss1 = self.byol_loss_per_sample(q1, z2)
        loss2 = self.byol_loss_per_sample(q2, z1)
        val_loss = ((loss1 + loss2) / 2).mean()

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    set_seed(42)
    pl.seed_everything(42, workers=True)

    train_loader, val_loader = dataset()
    model = BYOLModule()

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        callbacks=[ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min")],
        precision=16,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)

    # CUDA_VISIBLE_DEVICES=6 python main.py
