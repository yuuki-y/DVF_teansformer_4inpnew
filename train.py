import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import XrayTransformer
from dataset import XrayFolderDataset, DummyXrayDataset

# --- Utility Function for Saving Prediction ---
def save_prediction_as_nifti(tensor, output_path):
    """Saves a tensor prediction to a .nii.gz file."""
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Detach from graph, remove batch dim, and permute to (W, H, D, C)
    # Input tensor shape: (1, 3, 256, 256, 256)
    numpy_array = tensor.detach().squeeze(0).permute(3, 2, 1, 0).numpy()

    # Create a Nifti image
    affine = np.eye(4) # Standard identity affine
    nifti_img = nib.Nifti1Image(numpy_array, affine)

    # Save the image
    nib.save(nifti_img, output_path)
    print(f"Saved prediction to {output_path}")

# --- Main Execution Block ---
def main(args):
    # Setup
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")

    # For reproducibility
    torch.manual_seed(42)

    # Data
    if args.data_dir:
        print(f"Loading real data from {args.data_dir}")
        # Note: The real dataset assumes image and output sizes are 256
        # and doesn't use the --img-size or --out-size args.
        dataset = XrayFolderDataset(root_dir=args.data_dir)
    else:
        print("No data directory provided. Using dummy data generator.")
        dataset = DummyXrayDataset(
            num_samples=args.num_train_samples,
            img_size=args.img_size,
            out_size=args.out_size
        )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Model
    print("Initializing model...")
    model = XrayTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        decoder_start_res=args.decoder_start_res,
        out_channels=3,
        out_size=args.out_size,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params / 1e6:.2f}M parameters.")

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training Loop
    print("Starting training...")
    model.train()
    for epoch in range(args.epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

            # For demonstration, break after a few steps to keep it fast
            if args.demo_mode and i >= 5:
                break

    print("Training finished.")

    # Save Model
    model_save_path = os.path.join(args.output_dir, 'xray_transformer_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Inference and Save Example
    print("Running inference on one sample...")
    model.eval()
    with torch.no_grad():
        # Get one sample from the dataloader
        inputs, _ = next(iter(dataloader))
        inputs = inputs.to(device)

        # Run prediction
        prediction = model(inputs) # Shape: (B, 3, 256, 256, 256)

        # Save the first prediction in the batch
        prediction_save_path = os.path.join(args.output_dir, 'prediction_example.nii.gz')
        save_prediction_as_nifti(prediction, prediction_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train XrayTransformer model with dynamic data generation.")

    # --- Training Args ---
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--data-dir', type=str, default=None, help='Path to the root data directory. If not provided, a dummy dataset will be used.')
    parser.add_argument('--num-train-samples', type=int, default=50, help='Number of dummy training samples to generate per epoch.')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers for DataLoader.')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save model and predictions.')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA training.')
    parser.add_argument('--demo-mode', action='store_true', help='Run for only a few steps per epoch for quick demo.')

    # --- Model Hyperparameters ---
    parser.add_argument('--img-size', type=int, default=256, help='Input image size.')
    parser.add_argument('--out-size', type=int, default=256, help='Output image size.')
    # Use smaller, memory-friendly parameters for a quick test by default
    parser.add_argument('--patch-size', type=int, default=32, help='Patch size.') # Larger patch size -> smaller sequence
    parser.add_argument('--embed-dim', type=int, default=256, help='Embedding dimension.')
    parser.add_argument('--depth', type=int, default=2, help='Transformer depth (num layers).')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--mlp-ratio', type=float, default=2.0, help='MLP hidden dim ratio.')
    parser.add_argument('--decoder-start-res', type=int, default=16, help='Starting resolution of the 3D decoder.')

    args = parser.parse_args()

    # Add a flag for quick demo runs
    if args.demo_mode:
        print("--- RUNNING IN DEMO MODE ---")
        # In demo mode, drastically reduce sizes to prevent memory issues in limited environments
        args.img_size = 64
        args.patch_size = 8 # smaller patch size for smaller image
        args.out_size = 64
        # Calculate start_res based on out_size and num_upsamples (always 4 in this decoder)
        # out_size = start_res * 2^num_upsamples => start_res = out_size / 16
        args.decoder_start_res = args.out_size // 16
        args.epochs = 1
        args.num_train_samples = 10
        args.embed_dim=128
        args.depth=1
        args.num_heads=2

    main(args)
