import os
import sys
import glob
import numpy as np
import torch

from models.baseline_unet import BaselineSRUNet


def main():
    # Check command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find model weights
    possible_weights = [
        "models/best_model.pth",
        "weights/best_model.pth",
        "/content/weights/best_model.pth"
    ]

    weights_path = None
    for path in possible_weights:
        if os.path.exists(path):
            weights_path = path
            break

    if weights_path is None:
        print("ERROR: Model weights not found.")
        sys.exit(1)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model
    model = BaselineSRUNet().to(device)

    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Find all .npy files
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))

    print("Input files found:", len(input_files))

    if len(input_files) == 0:
        print("ERROR: No .npy files found in input directory.")
        sys.exit(1)

    # Process every file
    with torch.no_grad():
        for i, input_path in enumerate(input_files):

            filename = os.path.basename(input_path)

            # Load input
            noisy = np.load(input_path).astype(np.float32)

            # Clip input safely
            noisy = np.clip(noisy, 0.0, 1.0)

            # Convert to tensor: [1, 1, H, W]
            x = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)

            # Model prediction
            output = model(x)

            # Convert back to numpy
            restored = output.squeeze().detach().cpu().numpy()

            # Ensure correct output requirements
            restored = np.clip(restored, 0.0, 1.0)
            restored = np.nan_to_num(
                restored,
                nan=0.0,
                posinf=1.0,
                neginf=0.0
            ).astype(np.float32)

            # Save with SAME filename
            output_path = os.path.join(output_dir, filename)
            np.save(output_path, restored)

            print(
                f"[{i+1}/{len(input_files)}] "
                f"{filename} -> {restored.shape}"
            )

    print("\nInference completed successfully!")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
