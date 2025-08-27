import torch
import torch.nn as nn
import math
import argparse
import os
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- 1. Model Architecture ---
# The model architecture is based on the user's provided code.
# Key correction: The final Conv3d layer in the decoder now outputs 3 channels for (dx, dy, dz).

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class Transformer3DDecoder(nn.Module):
    """Decoder to reconstruct 3D image from sequence"""
    def __init__(self, embed_dim=768, num_patches_total=1024, decoder_start_res=16, out_channels=3, out_size=256):
        super().__init__()
        self.decoder_start_res = decoder_start_res
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.out_size = out_size

        self.proj = nn.Linear(embed_dim, (decoder_start_res**3) * embed_dim // 8)

        # Dynamically create decoder layers based on start and end resolution
        layers = []
        num_upsamples = int(math.log2(out_size // decoder_start_res))

        ch_in = embed_dim // 8
        ch_out = 256

        for i in range(num_upsamples):
            ch_out = ch_in // 2 if i > 0 else 256
            layers.append(nn.ConvTranspose3d(ch_in, ch_out, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm3d(ch_out))
            layers.append(nn.ReLU(True))
            ch_in = ch_out

        layers.append(nn.Conv3d(ch_in, self.out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.proj(x)
        x = x.view(x.size(0), self.embed_dim // 8, self.decoder_start_res, self.decoder_start_res, self.decoder_start_res)
        x = self.decoder(x)
        return x

class XrayTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=8, in_chans=1,
                 embed_dim=1024, depth=16, num_heads=16, mlp_ratio=8.,
                 decoder_start_res=16, out_channels=3, out_size=256):
        super().__init__()
        self.num_inputs = 4
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        num_patches_total = num_patches * self.num_inputs

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches_total, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(mlp_ratio * embed_dim), batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.decoder = Transformer3DDecoder(embed_dim, num_patches_total, decoder_start_res, out_channels, out_size)

    def forward(self, x):
        # x shape: (B, 4, 1, 256, 256)
        all_patches = [self.patch_embed(x[:, i, :, :, :]) for i in range(self.num_inputs)]
        x = torch.cat(all_patches, dim=1)
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        x = self.decoder(x) # Expected output: (B, 3, out_size, out_size, out_size)
        return x

# --- 2. Data Loading ---

class XrayFolderDataset(Dataset):
    """
    Dataset to load data from folders as specified in the request.
    It expects a root directory with the following structure:
    - root_dir/
      - frontal1/
      - frontal2/
      - lateral1/
      - lateral2/
      - target/
    It assumes that files are sorted alphabetically and correspond to each other.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.view_dirs = {
            'frontal1': os.path.join(root_dir, 'frontal1'),
            'frontal2': os.path.join(root_dir, 'frontal2'),
            'lateral1': os.path.join(root_dir, 'lateral1'),
            'lateral2': os.path.join(root_dir, 'lateral2'),
        }
        self.target_dir = os.path.join(root_dir, 'target')

        # Get the list of files, assuming they are consistently named and sorted
        self.file_list = sorted(os.listdir(self.view_dirs['frontal1']))

        # Basic validation
        for view, path in self.view_dirs.items():
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Directory not found: {path}")
        if not os.path.isdir(self.target_dir):
            raise FileNotFoundError(f"Directory not found: {self.target_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        # Load the 4 input images
        images = []
        for view in ['frontal1', 'frontal2', 'lateral1', 'lateral2']:
            img_path = os.path.join(self.view_dirs[view], filename)
            # Assuming .pt files, use torch.load
            images.append(torch.load(img_path))

        input_tensor = torch.stack(images) # Stacks to (4, 1, 256, 256)

        # Load the target .nii.gz file
        target_filename = filename.replace('.pt', '.nii.gz')
        target_path = os.path.join(self.target_dir, target_filename)
        nii_img = nib.load(target_path)
        # Convert to numpy array and then to tensor
        # Nibabel loads as (W, H, D, C), we need (C, D, H, W) for PyTorch
        target_array = nii_img.get_fdata(dtype=np.float32)
        target_tensor = torch.from_numpy(target_array).permute(3, 2, 1, 0)

        return input_tensor, target_tensor

class DummyXrayDataset(Dataset):
    """
    A dummy dataset that generates random data samples in memory.
    This is used to demonstrate the model's functionality without requiring
    a real dataset.
    """
    def __init__(self, num_samples=100, img_size=256, out_dims=3, out_size=256):
        self.num_samples = num_samples
        self.img_size = img_size
        self.out_dims = out_dims
        self.out_size = out_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Input: (4, 1, img_size, img_size)
        input_tensor = torch.rand(4, 1, self.img_size, self.img_size)
        # Target: (3, out_size, out_size, out_size)
        target_tensor = torch.rand(self.out_dims, self.out_size, self.out_size, self.out_size)
        return input_tensor, target_tensor

# --- 3. Utility Function for Saving Prediction ---
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

# --- 4. Main Execution Block ---
def main(args):
    # Setup
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")

    # For reproducibility
    torch.manual_seed(42)

    # Data
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
