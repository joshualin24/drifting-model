"""
Training script for Drifting Models on Galaxy10 DECaLS.
"""

import argparse
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# --- Assumes these modules exist in your folder as per your previous script ---
from model import DriftDiT_Tiny, DriftDiT_Small, DriftDiT_models
from drifting import (
    compute_V,
    normalize_features,
    normalize_drift,
)
from feature_encoder import create_feature_encoder
from utils import (
    EMA,
    WarmupLRScheduler,
    SampleQueue,
    save_checkpoint,
    load_checkpoint,
    save_image_grid,
    count_parameters,
    set_seed,
)

# --- Galaxy10 Configuration ---
GALAXY_CONFIG = {
    "dataset": "galaxy10",
    "model": "DriftDiT-Small",  # Increased capacity for complex textures
    "img_size": 32,             # Downsampled from 256 for faster training
    "in_channels": 3,
    "num_classes": 10,
    "batch_nc": 10,
    "batch_n_pos": 16,          # Smaller batch size for larger images
    "batch_n_neg": 16,
    "temperatures": [0.02, 0.05, 0.2],
    "lr": 2e-4,
    "weight_decay": 0.01,
    "grad_clip": 2.0,
    "ema_decay": 0.999,
    "warmup_steps": 2000,
    "epochs": 200,
    "alpha_min": 1.0,
    "alpha_max": 3.0,
    "use_feature_encoder": True, # Use pretrained ResNet features
    "queue_size": 128,
    "label_dropout": 0.1,
}


class Galaxy10Dataset(Dataset):
    """
    Galaxy10 DECaLS Dataset.
    10 classes of galaxies:
    0: Disturbed, 1: Merging, 2: Round Smooth, 3: In-between Smooth,
    4: Cigar Smooth, 5: Barred Spiral, 6: Unbarred Tight Spiral,
    7: Unbarred Loose Spiral, 8: Edge-on without Bulge, 9: Edge-on with Bulge
    """
    # url = "https://astro.utoronto.ca/~bovy/Galaxy10/Galaxy10_DECals.h5"
    url = "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5"
    filename = "Galaxy10_DECals.h5"

    def __init__(self, root, train=True, transform=None, download=True):
        self.root = root
        self.transform = transform
        self.filepath = os.path.join(root, self.filename)

        if download:
            self._download()

        if not os.path.exists(self.filepath):
            raise RuntimeError(f"Dataset not found at {self.filepath}")

        print(f"Loading Galaxy10 data from {self.filepath}...")
        # Load all data into memory (approx 2.5GB uncompressed)
        with h5py.File(self.filepath, 'r') as F:
            images = F['images'][:]  # Shape (17736, 256, 256, 3)
            labels = F['ans'][:]     # Shape (17736,)
        
        # Simple shuffle and split
        num_samples = len(labels)
        rng = np.random.RandomState(42)
        indices = rng.permutation(num_samples)
        split = int(0.8 * num_samples)
        
        if train:
            self.images = images[indices[:split]]
            self.labels = labels[indices[:split]]
        else:
            self.images = images[indices[split:]]
            self.labels = labels[indices[split:]]
            
        print(f"Loaded {len(self.labels)} images ({'train' if train else 'test'})")

    def _download(self):
        if os.path.exists(self.filepath):
            return
        os.makedirs(self.root, exist_ok=True)
        print(f"Downloading Galaxy10 DECaLS to {self.filepath}...")
        try:
            urllib.request.urlretrieve(self.url, self.filepath)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download: {e}")
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
            raise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx] # (256, 256, 3) numpy array
        label = self.labels[idx]

        # Transform expects PIL or Tensor. 
        # Since it's numpy uint8 [0-255], ToPILImage is safest start.
        if self.transform:
            img = self.transform(img)

        return img, int(label)


def sample_batch(queue: SampleQueue, num_classes: int, n_pos: int, device: torch.device) -> tuple:
    x_pos_list = []
    labels_list = []
    for c in range(num_classes):
        x_c = queue.sample(c, n_pos, device)
        x_pos_list.append(x_c)
        labels_list.append(torch.full((n_pos,), c, device=device, dtype=torch.long))
    return torch.cat(x_pos_list, dim=0), torch.cat(labels_list, dim=0)


def compute_drifting_loss(
    x_gen: torch.Tensor,
    labels_gen: torch.Tensor,
    x_pos: torch.Tensor,
    labels_pos: torch.Tensor,
    feature_encoder: Optional[nn.Module],
    temperatures: list,
) -> tuple:
    device = x_gen.device
    num_classes = labels_gen.max().item() + 1

    # Extract features (Multi-scale)
    if feature_encoder is None:
        feat_gen_list = [x_gen.flatten(start_dim=1)]
        feat_pos_list = [x_pos.flatten(start_dim=1)]
    else:
        feat_gen_maps = feature_encoder(x_gen)
        with torch.no_grad():
            feat_pos_maps = feature_encoder(x_pos)
        feat_gen_list = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feat_gen_maps]
        feat_pos_list = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feat_pos_maps]

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_drift_norm = 0.0
    num_losses = 0

    for c in range(num_classes):
        mask_gen = labels_gen == c
        mask_pos = labels_pos == c

        if not mask_gen.any() or not mask_pos.any():
            continue

        for scale_idx, (feat_gen, feat_pos) in enumerate(zip(feat_gen_list, feat_pos_list)):
            feat_gen_c = feat_gen[mask_gen]
            feat_pos_c = feat_pos[mask_pos]
            feat_neg_c = feat_gen_c # Negatives are generated samples (Algorithm 1)

            feat_gen_c_norm = F.normalize(feat_gen_c, p=2, dim=1)
            feat_pos_c_norm = F.normalize(feat_pos_c, p=2, dim=1)
            feat_neg_c_norm = F.normalize(feat_neg_c, p=2, dim=1)

            V_total = torch.zeros_like(feat_gen_c_norm)
            for tau in temperatures:
                V_tau = compute_V(feat_gen_c_norm, feat_pos_c_norm, feat_neg_c_norm, tau, mask_self=True)
                v_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
                V_total = V_total + V_tau / (v_norm + 1e-8)

            target = (feat_gen_c_norm + V_total).detach()
            loss_scale = F.mse_loss(feat_gen_c_norm, target)

            total_loss = total_loss + loss_scale
            total_drift_norm += (V_total ** 2).mean().item() ** 0.5
            num_losses += 1

    if num_losses == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {"loss": 0.0, "drift_norm": 0.0}

    loss = total_loss / num_losses
    info = {"loss": loss.item(), "drift_norm": total_drift_norm / num_losses}
    return loss, info


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    queue: SampleQueue,
    config: dict,
    device: torch.device,
    feature_encoder: Optional[nn.Module] = None,
) -> dict:
    model.train()
    num_classes = config["num_classes"]
    n_pos = config["batch_n_pos"]
    n_neg = config["batch_n_neg"]
    
    # Total batch size for generation
    batch_size = num_classes * n_neg
    labels = torch.arange(num_classes, device=device).repeat_interleave(n_neg)
    alpha = torch.empty(batch_size, device=device).uniform_(config["alpha_min"], config["alpha_max"])
    
    noise = torch.randn(batch_size, config["in_channels"], config["img_size"], config["img_size"], device=device)
    x_gen = model(noise, labels, alpha)
    x_pos, labels_pos = sample_batch(queue, num_classes, n_pos, device)

    loss, info = compute_drifting_loss(x_gen, labels, x_pos, labels_pos, feature_encoder, config["temperatures"])

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
    info["grad_norm"] = grad_norm.item()
    optimizer.step()

    return info


def fill_queue(queue: SampleQueue, dataloader: DataLoader, device: torch.device, min_samples: int = 64):
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x, labels = batch[0], batch[1]
        else:
            x, labels = batch, torch.zeros(batch.shape[0], dtype=torch.long)
        queue.add(x, labels)
        if queue.is_ready(min_samples):
            break


def train(output_dir: str = "./outputs/galaxy10", resume: Optional[str] = None, seed: int = 42):
    set_seed(seed)
    config = GALAXY_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transforms for Galaxy data
    # Note: We resize to 64x64 for this specific config
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config["img_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), # Galaxies are rotation invariant
        transforms.RandomRotation(180),  # Full rotation
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    print("Initializing Galaxy10 dataset...")
    train_dataset = Galaxy10Dataset(root="./data", train=True, download=True, transform=transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=128, # Larger batch size for queue filling
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Create model
    model_fn = DriftDiT_models[config["model"]]
    model = model_fn(
        img_size=config["img_size"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        label_dropout=config["label_dropout"],
    ).to(device)
    print(f"Model: {config['model']}, Parameters: {count_parameters(model):,}")

    ema = EMA(model, decay=config["ema_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95), weight_decay=config["weight_decay"])
    scheduler = WarmupLRScheduler(optimizer, warmup_steps=config["warmup_steps"], base_lr=config["lr"])
    
    queue = SampleQueue(
        num_classes=config["num_classes"],
        queue_size=config["queue_size"],
        sample_shape=(config["in_channels"], config["img_size"], config["img_size"]),
    )

    # Use Feature Encoder (ResNet)
    print("Creating ImageNet-pretrained feature encoder...")
    feature_encoder = create_feature_encoder(
        dataset="cifar10", # We use CIFAR settings as proxy for general RGB
        feature_dim=512,
        multi_scale=True,
        use_pretrained=True,
    ).to(device)
    feature_encoder.eval()
    for param in feature_encoder.parameters():
        param.requires_grad = False

    # Training Loop
    start_epoch = 0
    global_step = 0
    
    if resume:
        checkpoint = load_checkpoint(resume, model, ema, optimizer, scheduler)
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["step"]
        print(f"Resumed from epoch {start_epoch}")

    print(f"\nStarting training on Galaxy10...")
    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        fill_queue(queue, train_loader, device, min_samples=64)

        for batch_idx, (x_real, labels_real) in enumerate(train_loader):
            x_real, labels_real = x_real.to(device), labels_real.to(device)
            queue.add(x_real.cpu(), labels_real.cpu())

            if not queue.is_ready(config["batch_n_pos"]):
                continue

            info = train_step(model, optimizer, queue, config, device, feature_encoder)
            
            ema.update(model)
            scheduler.step()
            
            epoch_loss += info["loss"]
            num_batches += 1
            global_step += 1

            if global_step % 50 == 0:
                 print(f"Ep {epoch+1} | Step {global_step} | Loss: {info['loss']:.4f} | Drift: {info['drift_norm']:.4f} | LR: {scheduler.get_lr():.6f}")

        # Save checkpoint & samples
        if (epoch + 1) % 5 == 0:
            ckpt_path = output_dir / f"ckpt_ep{epoch+1}.pt"
            save_checkpoint(str(ckpt_path), model, ema, optimizer, scheduler, epoch, global_step, config)
            
            # Generate samples
            print("Generating samples...")
            ema.shadow.eval()
            samples = []
            for c in range(10):
                noise = torch.randn(8, 3, config["img_size"], config["img_size"], device=device)
                labels = torch.full((8,), c, device=device, dtype=torch.long)
                # CFG scale 1.5 usually good for generation
                with torch.no_grad():
                    x = ema.shadow.forward_with_cfg(noise, labels, alpha=1.5)
                samples.append(x)
            
            save_image_grid(torch.cat(samples), str(output_dir / f"samples_ep{epoch+1}.png"), nrow=8)
            print(f"Saved samples to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs/galaxy10")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(output_dir=args.output_dir, resume=args.resume, seed=args.seed)