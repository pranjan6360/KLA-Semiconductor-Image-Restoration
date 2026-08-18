import time
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from models.baseline_unet import BaselineSRUNet

def parse_args():
    parser = argparse.ArgumentParser(description="KLA Benchmark Evaluation Script for Semiconductor Image Restoration")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test .npy NoisyLR files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save restored .npy output files")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (.pth)")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size (default: 16)")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    
    test_p = Path(args.test_dir)
    output_p = Path(args.output_dir)
    weights_p = Path(args.weights)
    
    if not test_p.exists():
        raise FileNotFoundError(f"Test directory does not exist: {test_p}")
    if not weights_p.exists():
        raise FileNotFoundError(f"Model weights file does not exist: {weights_p}")
        
    output_p.mkdir(parents=True, exist_ok=True)
    
    # 1. Discover all test .npy files
    test_files = sorted(list(test_p.glob("*.npy")))
    total_images = len(test_files)
    if total_images == 0:
        raise RuntimeError(f"No .npy files found in test directory: {test_p}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print("KLA STANDALONE BENCHMARK EVALUATION")
    print("==================================================")
    print(f"Device: {device}")
    print(f"Test Directory: {test_p.resolve()}")
    print(f"Output Directory: {output_p.resolve()}")
    print(f"Weights Path: {weights_p.resolve()}")
    print(f"Found {total_images} test images.")
    
    # 2. Load Model Architecture & Weights
    model = BaselineSRUNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(weights_p, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 3. Batch-wise Inference Execution
    t_start = time.time()
    batch_size = args.batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, total_images, batch_size), desc="Restoring Test Images"):
            batch_files = test_files[i:i + batch_size]
            batch_tensors = []
            
            for file_path in batch_files:
                arr = np.load(file_path).astype(np.float32) # (128, 128)
                arr_clipped = np.clip(arr, 0.0, 1.0)
                tensor = torch.from_numpy(np.expand_dims(arr_clipped, axis=0)).float()
                batch_tensors.append(tensor)
                
            batch_in = torch.stack(batch_tensors, dim=0).to(device) # (B, 1, 128, 128)
            batch_out = model(batch_in) # (B, 1, 256, 256)
            
            batch_out_np = batch_out.cpu().numpy().astype(np.float32)
            
            for idx, file_path in enumerate(batch_files):
                restored_arr = batch_out_np[idx, 0] # (256, 256)
                restored_clipped = np.clip(restored_arr, 0.0, 1.0)
                
                # Save with exact same filename
                save_path = output_p / file_path.name
                np.save(save_path, restored_clipped)
                
    total_time = time.time() - t_start
    avg_time = total_time / total_images if total_images > 0 else 0.0
    
    print("\n==================================================")
    print("EVALUATION COMPLETE")
    print("==================================================")
    print(f"Total Restored Images: {total_images}")
    print(f"Output Resolution: (256, 256)")
    print(f"Output Data Type: float32")
    print(f"Total Inference Time: {total_time:.2f} seconds")
    print(f"Average Time Per Image: {avg_time * 1000.0:.2f} ms ({avg_time:.4f} s/img)")
    print(f"Outputs Saved To: {output_p.resolve()}")

if __name__ == '__main__':
    evaluate()
