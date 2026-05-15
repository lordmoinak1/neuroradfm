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
            sample = {k: os.path.join(data_path, subj, f"{subj}-{k}.nii.gz") for k in keys}
            sample["group"] = group
            subjects.append(sample)
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


class DINOHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=1024, bottleneck_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class DINO(pl.LightningModule):
    def __init__(self, out_dim=1024, bottleneck_dim=256, teacher_momentum=0.996, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.student_encoder = resnet10(spatial_dims=3, n_input_channels=4, feed_forward=False)
        self.student_head = DINOHead(512, out_dim, bottleneck_dim)

        self.teacher_encoder = resnet10(spatial_dims=3, n_input_channels=4, feed_forward=False)
        self.teacher_head = DINOHead(512, out_dim, bottleneck_dim)

        self.group_dro = GroupDROLoss(num_groups=4)
        self.register_buffer("center", torch.zeros(1, out_dim))

        for p_s, p_t in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False
        for p_s, p_t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        for ps, pt in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            pt.data = pt.data * self.hparams.teacher_momentum + ps.data * (1. - self.hparams.teacher_momentum)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data = pt.data * self.hparams.teacher_momentum + ps.data * (1. - self.hparams.teacher_momentum)

    def dino_loss_per_sample(self, student_out, teacher_out, temp_s=0.1, temp_t=0.04):
        student_out = student_out / temp_s
        teacher_out = F.softmax((teacher_out - self.center) / temp_t, dim=-1)
        log_probs = F.log_softmax(student_out, dim=-1)
        loss = -(teacher_out * log_probs).sum(dim=-1)
        return loss  # shape: (B,)

    def forward(self, x):
        return self.student_head(self.student_encoder(x))

    def training_step(self, batch, batch_idx):
        x1 = torch.cat([batch["view1"]["t1c"], batch["view1"]["t1n"],
                        batch["view1"]["t2f"], batch["view1"]["t2w"]], dim=1)
        x2 = torch.cat([batch["view2"]["t1c"], batch["view2"]["t1n"],
                        batch["view2"]["t2f"], batch["view2"]["t2w"]], dim=1)

        s_out1 = self.student_head(self.student_encoder(x1))
        s_out2 = self.student_head(self.student_encoder(x2))

        with torch.no_grad():
            self.update_teacher()
            t_out1 = self.teacher_head(self.teacher_encoder(x1))
            t_out2 = self.teacher_head(self.teacher_encoder(x2))

        loss1 = self.dino_loss_per_sample(s_out1, t_out2)
        loss2 = self.dino_loss_per_sample(s_out2, t_out1)

        all_losses = (loss1 + loss2) / 2
        group_ids = batch["view1"]["group"]

        loss = self.group_dro(all_losses, group_ids)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # update center
        with torch.no_grad():
            batch_center = torch.cat([t_out1, t_out2], dim=0).mean(dim=0, keepdim=True)
            self.center = 0.9 * self.center + 0.1 * batch_center

        return loss

    def validation_step(self, batch, batch_idx):
        x = torch.cat([batch["view1"]["t1c"], batch["view1"]["t1n"],
                       batch["view1"]["t2f"], batch["view1"]["t2w"]], dim=1)
        s_out = self(x)
        val_loss = -F.softmax(s_out, dim=-1).max(dim=-1)[0].mean()
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    set_seed(42)
    pl.seed_everything(42, workers=True)

    train_loader, val_loader = dataset()
    model = DINO()

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        callbacks=[ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min")],
        precision=16,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)

    # CUDA_VISIBLE_DEVICES=1 python3 main.py
