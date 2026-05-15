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


def dataset():
    keys = ['t1c', 't1n', 't2f', 't2w']
    base_transforms = [
        transforms.LoadImaged(keys=keys),
        transforms.EnsureChannelFirstd(keys=keys),
        transforms.Orientationd(keys=keys, axcodes="RAS"),
        transforms.Spacingd(keys=keys, pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
        transforms.NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        transforms.ResizeWithPadOrCropd(keys=keys, spatial_size=(128, 128, 128)),
        transforms.ToTensord(keys=keys)
    ]
    transform = transforms.Compose(base_transforms)

    def generate_splits(data_path):
        subjects = []
        for subj in os.listdir(data_path):
            if not os.path.isdir(os.path.join(data_path, subj)):
                continue
            if 'GLI' in subj or 'UPENN' in subj or 'UCSF' in subj or 'Patient' in subj:
                group = 0
            elif 'MET' in subj:
                group = 1
            elif 'PED' in subj:
                group = 2
            elif 'MEN' in subj:
                group = 3
            else:
                group = -1

            item = {k: os.path.join(data_path, subj, f"{subj}-{k}.nii.gz") for k in keys}
            item['group'] = group
            subjects.append(item)
        return subjects

    train_data = generate_splits('/path/to/train/')
    val_data = generate_splits('/path/to/val/')

    train_ds = Dataset(train_data, transform=transform)
    val_ds = Dataset(val_data, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4,
                              collate_fn=pad_list_data_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4,
                            collate_fn=pad_list_data_collate, drop_last=True)
    return train_loader, val_loader


class GroupDROLoss(nn.Module):
    def __init__(self, num_groups=4):
        super().__init__()
        self.num_groups = num_groups

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


class MAE3D(pl.LightningModule):
    def __init__(self, mask_ratio=0.75, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = resnet10(spatial_dims=3, n_input_channels=4, feed_forward=True)
        self.decoder = nn.Sequential(
            nn.Linear(400, 256),
            nn.ReLU(),
            nn.Linear(256, 400)
        )

        self.group_dro = GroupDROLoss(num_groups=4)

    def mask_input(self, x, mask_ratio):
        B, C, D, H, W = x.shape
        num_voxels = D * H * W
        num_mask = int(mask_ratio * num_voxels)

        mask = torch.zeros(B, num_voxels, device=x.device)
        for i in range(B):
            perm = torch.randperm(num_voxels, device=x.device)
            mask[i, perm[:num_mask]] = 1
        mask = mask.view(B, 1, D, H, W).expand(-1, C, -1, -1, -1)

        x_masked = x.clone()
        x_masked[mask.bool()] = 0
        return x_masked, mask

    def forward(self, x):
        x_masked, mask = self.mask_input(x, self.hparams.mask_ratio)
        features = self.encoder(x_masked)
        recon = self.decoder(features)
        return recon, x, mask

    def mae_loss_per_sample(self, recon, target):
        with torch.no_grad():
            target_features = self.encoder(target)
        return F.mse_loss(recon, target_features, reduction="none").mean(dim=1)  # (B,)

    def training_step(self, batch, batch_idx):
        x = torch.cat([batch["t1c"], batch["t1n"], batch["t2f"], batch["t2w"]], dim=1)
        recon, target, _ = self.forward(x)
        per_sample_loss = self.mae_loss_per_sample(recon, target)
        group_ids = batch["group"]
        loss = self.group_dro(per_sample_loss, group_ids)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = torch.cat([batch["t1c"], batch["t1n"], batch["t2f"], batch["t2w"]], dim=1)
        recon, target, _ = self.forward(x)
        with torch.no_grad():
            val_loss = F.mse_loss(self.encoder(target), recon)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    set_seed(42)
    pl.seed_everything(42, workers=True)

    train_loader, val_loader = dataset()
    model = MAE3D(mask_ratio=0.75)

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        callbacks=[ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min")],
        precision=16,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)

    # CUDA_VISIBLE_DEVICES=7 python main.py
