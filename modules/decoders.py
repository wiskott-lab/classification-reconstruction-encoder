import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class DecoderFCSmall(nn.Module):
    # as from the antagonist paper
    def __init__(self, latent_dim=512):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=latent_dim, out_features=784)
        self.tanh_1 = nn.Tanh()
        self.unflatten_1 = nn.Unflatten(dim=1, unflattened_size=(1, 28, 28))

    def forward(self, x):
        x = self.unflatten_1(self.tanh_1(self.linear_1(x)))
        return x


class DecoderFCMedium(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=latent_dim, out_features=616)
        self.relu_1 = nn.ReLU()

        self.linear_2 = nn.Linear(in_features=616, out_features=784)
        self.tanh_2 = nn.Tanh()
        self.unflatten_2 = nn.Unflatten(dim=1, unflattened_size=(1, 28, 28))

    def forward(self, x):
        x = self.relu_1(self.linear_1(x))
        x = self.unflatten_2(self.tanh_2(self.linear_2(x)))
        return x


class DecoderFCLarge(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=latent_dim, out_features=616)
        self.relu_1 = nn.ReLU()

        self.linear_2 = nn.Linear(in_features=616, out_features=684)
        self.relu_2 = nn.ReLU()

        self.linear_3 = nn.Linear(in_features=684, out_features=716)
        self.relu_3 = nn.ReLU()

        self.linear_4 = nn.Linear(in_features=716, out_features=784)
        self.tanh_4 = nn.Tanh()
        self.unflatten_4 = nn.Unflatten(dim=1, unflattened_size=(1, 28, 28))

    def forward(self, x):
        x = self.relu_1(self.linear_1(x))
        x = self.relu_2(self.linear_2(x))
        x = self.relu_3(self.linear_3(x))
        x = self.unflatten_4(self.tanh_4(self.linear_4(x)))
        return x


class DecoderCNNSmall(nn.Module):

    def __init__(self):
        # Decoder according to https://github.com/ternaus/TernausNet/blob/master/ternausnet/models.py
        super().__init__()
        self.conv_t_1 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_1 = nn.ReLU()  # 16x8x8

        self.conv_t_2 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=2, output_padding=1)
        self.relu_2 = nn.ReLU()  # 16x14x14

        self.conv_t_3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tanh_3 = nn.Tanh()  # 1x28x28

    def forward(self, x):
        x = self.relu_1(self.conv_t_1(x))
        x = self.relu_2(self.conv_t_2(x))
        x = self.tanh_3(self.conv_t_3(x))
        return x


class DecoderCNNLarge(nn.Module):

    def __init__(self):
        # Decoder according to https://github.com/ternaus/TernausNet/blob/master/ternausnet/models.py
        super().__init__()
        self.conv_t_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_1 = nn.ReLU()  # 128x8x8

        self.conv_t_2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_2 = nn.ReLU()  # 128x16x16

        self.conv_t_3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_3 = nn.ReLU()  # 64x32x32

        self.conv_t_4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.relu_4 = nn.ReLU()  # 64x32x32

        self.conv_t_5 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.tanh_5 = nn.Tanh()  # 3x32x32

    def forward(self, x):
        x = self.relu_1(self.conv_t_1(x))
        x = self.relu_2(self.conv_t_2(x))
        x = self.relu_3(self.conv_t_3(x))
        x = self.relu_4(self.conv_t_4(x))
        x = self.tanh_5(self.conv_t_5(x))
        return x


class DecoderViT(nn.Module):  # adapted from https://github.com/facebookresearch/mae

    def __init__(self, embed_dim, decoder_embed_dim, num_patches, decoder_num_heads, mlp_ratio, decoder_depth,
                 patch_size, in_chans):
        super().__init__()
        norm_layer = nn.LayerNorm
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]
        # cls_output = x[:, 0, :]
        return x
