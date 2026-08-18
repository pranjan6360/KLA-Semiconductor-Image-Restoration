import os
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from dataset import get_dataloaders
from models.baseline_unet import BaselineSRUNet
from utils.losses import CombinedLoss
from utils.metrics import calculate_psnr, calculate_ssim

def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline 2x SR-UNet for KLA Semiconductor Restoration")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for AdamW (default: 0.0001)")
    parser.add_argument("--data_dir", type=str, default="../data_sets", help="Path to data_sets directory")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory to save model weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    
    for noisy, gt, _ in dataloader:
        noisy = noisy.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                pred = model(noisy)
                loss, _, _ = criterion(pred, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(noisy)
            loss, _, _ = criterion(pred, gt)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item() * noisy.size(0)
        
    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    psnr_list = []
    ssim_list = []
    
    with torch.no_grad():
        for noisy, gt, _ in dataloader:
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            
            if torch.cuda.is_available():
                with autocast():
                    pred = model(noisy)
                    loss, _, _ = criterion(pred, gt)
            else:
                pred = model(noisy)
                loss, _, _ = criterion(pred, gt)
                
            total_loss += loss.item() * noisy.size(0)
            
            # Calculate PSNR & SSIM metrics batch-wise
            pred_np = pred.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()
            
            for b in range(pred_np.shape[0]):
                p_img = pred_np[b, 0]
                g_img = gt_np[b, 0]
                
                psnr_val = calculate_psnr(p_img, g_img, data_range=1.0)
                ssim_val = calculate_ssim(p_img, g_img, data_range=1.0)
                
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                
    avg_loss = total_loss / len(dataloader.dataset)
    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))
    
    return avg_loss, avg_psnr, avg_ssim


def main():
    args = parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== KLA PS01 RESTORATION MODEL TRAINING ===")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Learning Rate: {args.lr}")
    
    weights_path = Path(args.weights_dir)
    weights_path.mkdir(parents=True, exist_ok=True)
    best_model_save_path = weights_path / "best_model.pth"
    
    # Load Data
    train_loader, val_loader = get_dataloaders(
        data_root=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        train_augment=True
    )
    print(f"Training pairs: {len(train_loader.dataset)} | Validation pairs: {len(val_loader.dataset)}")
    
    # Model, Loss, Optimizer, Scaler
    model = BaselineSRUNet(in_channels=1, out_channels=1).to(device)
    criterion = CombinedLoss(charbonnier_eps=1e-3, ssim_weight=0.2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler() if device.type == "cuda" else None
    
    best_val_psnr = -float('inf')
    
    print("\nStarting Training Loop...")
    print("-" * 80)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val PSNR (dB)':<14} | {'Val SSIM':<10} | {'Time (s)':<8}")
    print("-" * 80)
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
        
        elapsed = time.time() - t0
        
        print(f"{epoch:<8} | {train_loss:<12.6f} | {val_loss:<12.6f} | {val_psnr:<14.2f} | {val_ssim:<10.4f} | {elapsed:<8.2f}")
        
        # Save best model checkpoint based on Validation PSNR
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
            }, best_model_save_path)
            
    print("-" * 80)
    print(f"Training Completed successfully!")
    print(f"Best Validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Saved best model weights to: {best_model_save_path.resolve()}")

if __name__ == '__main__':
    main()
