import argparse
from pathlib import Path
import numpy as np
import torch

from models.baseline_unet import BaselineSRUNet

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single .npy NoisyLR image")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npy NoisyLR file")
    parser.add_argument("--output", type=str, required=True, help="Path to save restored output .npy file")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (.pth)")
    return parser.parse_args()


def run_inference(input_path, output_path, weights_path):
    input_p = Path(input_path)
    output_p = Path(output_path)
    weights_p = Path(weights_path)
    
    if not input_p.exists():
        raise FileNotFoundError(f"Input file not found: {input_p}")
    if not weights_p.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_p}")
        
    output_p.parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model Weights
    model = BaselineSRUNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(weights_p, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 2. Load .npy file using numpy.load()
    input_arr = np.load(input_p).astype(np.float32) # (128, 128)
    
    # 3. Clip input to [0, 1]
    input_arr_clipped = np.clip(input_arr, 0.0, 1.0)
    
    # 4. Convert to Tensor (1, 1, 128, 128)
    tensor_in = torch.from_numpy(np.expand_dims(np.expand_dims(input_arr_clipped, axis=0), axis=0)).float().to(device)
    
    # 5. Run Model Inference
    with torch.no_grad():
        tensor_out = model(tensor_in) # (1, 1, 256, 256)
        
    # 6. Post-process to (256, 256) float32 in [0, 1]
    out_arr = tensor_out.squeeze().cpu().numpy().astype(np.float32)
    out_arr_clipped = np.clip(out_arr, 0.0, 1.0)
    
    # 7. Save as .npy
    np.save(output_p, out_arr_clipped)
    
    print(f"Successfully processed: {input_p.name}")
    print(f"  Input shape: {input_arr.shape} | Output shape: {out_arr_clipped.shape}")
    print(f"  Output dtype: {out_arr_clipped.dtype} | Output range: [{out_arr_clipped.min():.4f}, {out_arr_clipped.max():.4f}]")
    print(f"  Saved restored array to: {output_p.resolve()}")

def main():
    args = parse_args()
    run_inference(args.input, args.output, args.weights)

if __name__ == '__main__':
    main()
