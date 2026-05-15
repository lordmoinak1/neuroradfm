#!/usr/bin/env python3
import os
import torch
import numpy as np
from tqdm import tqdm

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, ResizeWithPadOrCropd, ToTensord
)
from monai.data import Dataset, DataLoader, pad_list_data_collate

from byol_dro.main import BYOLModule
from dino_dro.main import DINO
from moco_dro.main import MoCoModule
from mae_dro.main import MAE3D


def get_infer_loader(data_dir, batch_size=1):
    infer_transforms = Compose([
        LoadImaged(keys=['t1c', 't1n', 't2f', 't2w']),
        EnsureChannelFirstd(keys=['t1c', 't1n', 't2f', 't2w']),
        Orientationd(keys=['t1c', 't1n', 't2f', 't2w'], axcodes="RAS"),
        Spacingd(keys=['t1c', 't1n', 't2f', 't2w'], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
        NormalizeIntensityd(keys=['t1c', 't1n', 't2f', 't2w'], nonzero=True, channel_wise=True),
        ResizeWithPadOrCropd(keys=['t1c', 't1n', 't2f', 't2w'], spatial_size=(128, 128, 128)),
        ToTensord(keys=['t1c', 't1n', 't2f', 't2w'])
    ])

    def get_subjects(data_path):
        subjects = []
        for i in os.listdir(data_path):
            subj = {
                't1c': os.path.join(data_path, i, f'{i}-t1c.nii.gz'),
                't1n': os.path.join(data_path, i, f'{i}-t1n.nii.gz'),
                't2f': os.path.join(data_path, i, f'{i}-t2f.nii.gz'),
                't2w': os.path.join(data_path, i, f'{i}-t2w.nii.gz'),
                'id': i
            }
            subjects.append(subj)
        return subjects

    subjects = get_subjects(data_dir)
    dataset = Dataset(data=subjects, transform=infer_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=pad_list_data_collate)
    return loader, subjects

def _ensure_vec(feat: torch.Tensor) -> torch.Tensor:
    """
    Make sure features are [B, C] by pooling spatial dims if present.
    Accepts shapes like [B, C], [B, C, D], [B, C, H, W], [B, C, D, H, W].
    """
    if feat.ndim == 2:
        return feat
    # pool over all dims after channel
    while feat.ndim > 2:
        feat = torch.mean(feat, dim=-1)  # iterative mean over last dim
    return feat  # [B, C]

def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

class EnsembleEncoders(torch.nn.Module):
    """
    Wraps BYOL, DINO, MAE, MoCo encoders and returns an L2-normalized concat vector:
        z = [norm(z_byol) || norm(z_dino) || norm(z_mae) || norm(z_moco)]
    """
    def __init__(self, ckpt_byol: str, ckpt_dino: str, ckpt_mae: str, ckpt_moco: str, device: str = "cuda"):
        super().__init__()
        self.device = device

        # Load checkpoints
        self.byol = BYOLModule.load_from_checkpoint(ckpt_byol, strict=False).eval().to(device)
        self.dino = DINO.load_from_checkpoint(ckpt_dino, strict=False).eval().to(device)
        self.mae  = MAE3D.load_from_checkpoint(ckpt_mae, strict=False).eval().to(device)
        self.moco = MoCoModule.load_from_checkpoint(ckpt_moco, strict=False).eval().to(device)

    @torch.no_grad()
    def forward(self, x4: torch.Tensor):
        """
        x4: [B, 4, 128, 128, 128] tensor (modalities concatenated along channel)
        returns:
            z_fused: [B, Dsum]
            parts: dict of per-model embeddings (all [B, D_i])
        """
        # BYOL
        z_byol = self.byol.online_encoder(x4)
        z_byol = _ensure_vec(z_byol)
        z_byol = _l2norm(z_byol)

        # DINO (encoder only; no head)
        z_dino = self.dino.student_encoder(x4)
        z_dino = _ensure_vec(z_dino)
        z_dino = _l2norm(z_dino)

        # MAE (encoder output)
        z_mae = self.mae.encoder(x4)
        z_mae = _ensure_vec(z_mae)
        z_mae = _l2norm(z_mae)

        # MoCo (encoder_q; not the projection head)
        z_moco = self.moco.encoder_q(x4)
        z_moco = _ensure_vec(z_moco)
        z_moco = _l2norm(z_moco)

        z_fused = torch.cat([z_byol, z_dino, z_mae, z_moco], dim=-1)
        parts = {
            "byol": z_byol,
            "dino": z_dino,
            "mae":  z_mae,
            "moco": z_moco,
        }
        return z_fused, parts

def run_inference_ensemble(
    ckpt_byol: str,
    ckpt_dino: str,
    ckpt_mae: str,
    ckpt_moco: str,
    data_dir: str,
    output_dir: str,
    save_parts: bool = True
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EnsembleEncoders(
        ckpt_byol=ckpt_byol,
        ckpt_dino=ckpt_dino,
        ckpt_mae=ckpt_mae,
        ckpt_moco=ckpt_moco,
        device=device
    )

    loader, subjects = get_infer_loader(data_dir)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            subj = subjects[i]

            x = torch.cat([batch["t1c"], batch["t1n"], batch["t2f"], batch["t2w"]], dim=1).to(device)
            z_fused, parts = model(x)  # [B, Dsum], dict

            # Assume batch_size = 1
            fused_np = z_fused[0].cpu().numpy()
            np.save(os.path.join(output_dir, f"{subj['id']}.npy"), fused_np)

if __name__ == "__main__":
    # ----- CUMC -----
    run_inference_ensemble(
        ckpt_byol="/home/moinak/project_neuroradiology_3d/foundation_model/src/byol_dro/lightning_logs/version_0/checkpoints/epoch=3-step=2444.ckpt",
        ckpt_dino="/home/moinak/project_neuroradiology_3d/foundation_model/src/dino_dro/lightning_logs/version_0/checkpoints/epoch=61-step=37882.ckpt",
        ckpt_mae ="/home/moinak/project_neuroradiology_3d/foundation_model/src/mae_dro/lightning_logs/version_0/checkpoints/epoch=80-step=37098.ckpt",
        ckpt_moco="/home/moinak/project_neuroradiology_3d/foundation_model/src/moco_dro/lightning_logs/version_0/checkpoints/epoch=99-step=45800.ckpt",
        data_dir="/home/moinak/datasets/brain/cumc/cleaned_ants/",
        output_dir="/home/moinak/project_neuroradiology_3d/foundation_model/applications/feature_embeddings/ensemble/cumc/",
        save_parts=False
    )

    # ----- UCSF Test -----
    run_inference_ensemble(
        ckpt_byol="/home/moinak/project_neuroradiology_3d/foundation_model/src/byol_dro/lightning_logs/version_0/checkpoints/epoch=3-step=2444.ckpt",
        ckpt_dino="/home/moinak/project_neuroradiology_3d/foundation_model/src/dino_dro/lightning_logs/version_0/checkpoints/epoch=61-step=37882.ckpt",
        ckpt_mae ="/home/moinak/project_neuroradiology_3d/foundation_model/src/mae_dro/lightning_logs/version_0/checkpoints/epoch=80-step=37098.ckpt",
        ckpt_moco="/home/moinak/project_neuroradiology_3d/foundation_model/src/moco_dro/lightning_logs/version_0/checkpoints/epoch=99-step=45800.ckpt",
        data_dir="/home/moinak/datasets/brain/nofm_dataset/ucsf_test/",
        output_dir="/home/moinak/project_neuroradiology_3d/foundation_model/applications/feature_embeddings/ensemble/ucsf_test/",
        save_parts=False
    )

    # ----- UPenn Test -----
    run_inference_ensemble(
        ckpt_byol="/home/moinak/project_neuroradiology_3d/foundation_model/src/byol_dro/lightning_logs/version_0/checkpoints/epoch=3-step=2444.ckpt",
        ckpt_dino="/home/moinak/project_neuroradiology_3d/foundation_model/src/dino_dro/lightning_logs/version_0/checkpoints/epoch=61-step=37882.ckpt",
        ckpt_mae ="/home/moinak/project_neuroradiology_3d/foundation_model/src/mae_dro/lightning_logs/version_0/checkpoints/epoch=80-step=37098.ckpt",
        ckpt_moco="/home/moinak/project_neuroradiology_3d/foundation_model/src/moco_dro/lightning_logs/version_0/checkpoints/epoch=99-step=45800.ckpt",
        data_dir="/home/moinak/datasets/brain/nofm_dataset/upenn_test/",
        output_dir="/home/moinak/project_neuroradiology_3d/foundation_model/applications/feature_embeddings/ensemble/upenn_test/",
        save_parts=False
    )
