import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from monai import transforms
from monai.data import (
    Dataset,
    DataLoader,
    pad_list_data_collate
    )
from monai.utils import set_determinism
from monai.networks.nets import resnet10, ViTAutoEnc

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
    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=['t1c', 't1n', 't2f', 't2w']),
            transforms.EnsureChannelFirstd(keys=['t1c', 't1n', 't2f', 't2w']),
            transforms.Orientationd(keys=['t1c', 't1n', 't2f', 't2w'], axcodes="RAS"),
            transforms.Spacingd(keys=['t1c', 't1n', 't2f', 't2w'], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
            transforms.NormalizeIntensityd(keys=['t1c', 't1n', 't2f', 't2w'], nonzero=True, channel_wise=True),
            transforms.ResizeWithPadOrCropd(keys=['t1c', 't1n', 't2f', 't2w'], spatial_size=(128, 128, 128)),
            transforms.RandSpatialCropd(keys=['t1c', 't1n', 't2f', 't2w'], roi_size=(128, 128, 128), random_center=True),
            transforms.RandGaussianNoised(keys=['t1c', 't1n', 't2f', 't2w'], prob=0.2),
            transforms.ToTensord(keys=['t1c', 't1n', 't2f', 't2w'])
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=['t1c', 't1n', 't2f', 't2w']),
            transforms.EnsureChannelFirstd(keys=['t1c', 't1n', 't2f', 't2w']),
            transforms.Orientationd(keys=['t1c', 't1n', 't2f', 't2w'], axcodes="RAS"),
            transforms.Spacingd(keys=['t1c', 't1n', 't2f', 't2w'], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
            transforms.NormalizeIntensityd(keys=['t1c', 't1n', 't2f', 't2w'], nonzero=True, channel_wise=True),
            transforms.ResizeWithPadOrCropd(keys=['t1c', 't1n', 't2f', 't2w'], spatial_size=(128, 128, 128)),
            transforms.RandSpatialCropd(keys=['t1c', 't1n', 't2f', 't2w'], roi_size=(128, 128, 128), random_center=True),
            transforms.ToTensord(keys=['t1c', 't1n', 't2f', 't2w'])
        ]
    )

    def generate_splits(data_path):
        subjects = []
        for i in os.listdir(data_path):
            if 'GLI' in i or 'UPENN' in i or 'UCSF' in i or 'Patient' in i:
                group = 0
            if 'MET' in i:
                group = 1
            if 'PED' in i:
                group = 2
            if 'MEN' in i:
                group = 3
            subject = {
                't1c': os.path.join(data_path+i, i+'-t1c.nii.gz'),
                't1n': os.path.join(data_path+i, i+'-t1n.nii.gz'),
                't2f': os.path.join(data_path+i, i+'-t2f.nii.gz'),
                't2w': os.path.join(data_path+i, i+'-t2w.nii.gz'),
                'group': group
                }
            subjects.append(subject)
        return subjects
    
    train_subjects = generate_splits('/path/to/train/')
    val_subjects = generate_splits('/path/to/val/')

    dataset = Dataset(data=train_subjects, transform=train_transforms) #, cache_num=24, cache_rate=1, num_workers=2)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=pad_list_data_collate, drop_last=True)

    dataset = Dataset(data=val_subjects, transform=val_transforms) #, cache_num=24, cache_rate=1, num_workers=2)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=pad_list_data_collate, drop_last=True)

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

        # Stack group losses and only backprop through the max
        group_losses = torch.stack(group_losses)
        max_loss = group_losses.max()
        return max_loss


class MoCoModule(pl.LightningModule):
    def __init__(self, feature_dim=128, K=65536, m=0.999, T=0.07, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_q = resnet10(spatial_dims=3, n_input_channels=4, feed_forward=False)
        self.encoder_k = resnet10(spatial_dims=3, n_input_channels=4, feed_forward=False)

        self.fc_q = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim)
        )
        self.fc_k = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim)
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.fc_q.parameters(), self.fc_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(feature_dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.group_dro = GroupDROLoss(num_groups=4)

    @torch.no_grad()
    def momentum_update(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.hparams.m + param_q.data * (1. - self.hparams.m)
        for param_q, param_k in zip(self.fc_q.parameters(), self.fc_k.parameters()):
            param_k.data = param_k.data * self.hparams.m + param_q.data * (1. - self.hparams.m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        K = self.hparams.K
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % K

    def forward(self, x):
        return self.fc_q(self.encoder_q(x))

    def training_step(self, batch, batch_idx):
        im_q = torch.cat([batch["t1c"], batch["t1n"], batch["t2f"], batch["t2w"]], dim=1)
        im_k = im_q + 0.01 * torch.randn_like(im_q)

        q = F.normalize(self.fc_q(self.encoder_q(im_q)), dim=1)
        with torch.no_grad():
            self.momentum_update()
            k = F.normalize(self.fc_k(self.encoder_k(im_k)), dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.hparams.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # shape (N,)

        group_ids = torch.tensor(batch["group"], device=self.device)
        loss = self.group_dro(ce_loss, group_ids)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.dequeue_and_enqueue(k)
        return loss

    def validation_step(self, batch, batch_idx):
        im_q = torch.cat([batch["t1c"], batch["t1n"], batch["t2f"], batch["t2w"]], dim=1)
        im_k = im_q + 0.01 * torch.randn_like(im_q)

        q = F.normalize(self.fc_q(self.encoder_q(im_q)), dim=1)
        k = F.normalize(self.fc_k(self.encoder_k(im_k)), dim=1)
        val_loss = -F.cosine_similarity(q, k, dim=1).mean()
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

@torch.no_grad()
def concat_all_gather(tensor):
    """Gather tensors from all GPUs (no gradient)."""
    if torch.distributed.is_initialized():
        tensors_gather = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        return torch.cat(tensors_gather, dim=0)
    return tensor


if __name__ == "__main__":
    set_seed(42)
    
    train_loader, val_loader = dataset()

    model = MoCoModule()

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        callbacks=[ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min")],
        precision=16,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)

    # CUDA_VISIBLE_DEVICES=0 python3 main.py
