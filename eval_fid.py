import torch
import os
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from pytorch_fid import fid_score

# 載入你訓練腳本中的模型定義與類別
from model import DriftDiT_models
from utils import load_checkpoint
from train_galaxy import Galaxy10Dataset  # 確保 train.py 在同一個目錄

def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" usando device: {device}")

    # --- 1. 準備路徑 ---
    output_path = Path(args.eval_dir)
    real_path = output_path / "real_samples"
    fake_path = output_path / "fake_samples"
    
    for p in [real_path, fake_path]:
        if p.exists(): shutil.rmtree(p)
        p.mkdir(parents=True)

    # --- 2. 準備真實數據集 (Reference) ---
    print("📦 Extracting real samples from Galaxy10...")
    transform_real = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # 這裡載入測試集或驗證集來作為對比基準
    dataset = Galaxy10Dataset(root="./data", train=False, transform=transform_real)
    num_eval = min(len(dataset), args.num_samples)
    
    for i in tqdm(range(num_eval), desc="Saving Real Images"):
        img_tensor, _ = dataset[i]
        # 反標準化
        img = transforms.ToPILImage()(img_tensor * 0.5 + 0.5)
        img.save(real_path / f"real_{i}.png")

    # --- 3. 載入訓練好的模型 ---
    print(f"🚀 Loading model from {args.checkpoint}...")
    model_fn = DriftDiT_models[args.model_type]
    model = model_fn(
        img_size=args.img_size,
        in_channels=3,
        num_classes=10
    ).to(device)

    # 載入 checkpoint (主要提取 EMA 權重)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print("✅ EMA weights loaded successfully.")
    else:
        model.load_state_dict(ckpt["model"])
        print("⚠️ EMA not found, using raw model weights.")
    
    model.eval()

    # --- 4. 生成樣本 ---
    print(f"🎨 Generating {args.num_samples} fake samples...")
    batch_size = args.batch_size
    generated_count = 0
    
    pbar = tqdm(total=args.num_samples, desc="Generating Images")
    while generated_count < args.num_samples:
        curr_batch = min(batch_size, args.num_samples - generated_count)
        
        # 隨機或均勻抽取星系類別 (0-9)
        labels = torch.randint(0, 10, (curr_batch,), device=device)
        noise = torch.randn(curr_batch, 3, args.img_size, args.img_size, device=device)
        
        with torch.no_grad():
            # 使用 CFG Scale (推薦 1.5 - 2.0 提升品質)
            samples = model.forward_with_cfg(noise, labels, alpha=args.cfg_scale)
            samples = (samples * 0.5 + 0.5).clamp(0, 1)
        
        for j in range(samples.size(0)):
            img = transforms.ToPILImage()(samples[j].cpu())
            img.save(fake_path / f"gen_{generated_count}.png")
            generated_count += 1
        
        pbar.update(curr_batch)
    pbar.close()

    # --- 5. 計算 FID ---
    print(f"📊 Calculating FID between {real_path} and {fake_path}...")
    fid_value = fid_score.calculate_fid_given_paths(
        paths=[str(real_path), str(fake_path)],
        batch_size=args.batch_size,
        device=device,
        dims=2048
    )

    print("\n" + "="*40)
    print(f"🏆 FINAL FID SCORE: {fid_value:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ckpt_epX.pt")
    parser.add_argument("--model_type", type=str, default="DriftDiT-Small")
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=2048, help="FID is more accurate with > 2048 images")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--eval_dir", type=str, default="./fid_evaluation")
    
    args = parser.parse_args()
    run_evaluation(args)