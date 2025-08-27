import torch
import torch.nn as nn
import math

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
