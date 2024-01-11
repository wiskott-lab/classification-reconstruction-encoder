import torch
import torch.nn as nn
from torch.nn.functional import softplus
from timm.models.vision_transformer import PatchEmbed, Block


class EncoderFCSmall(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.flatten_1 = nn.Flatten(start_dim=1)
        self.linear_1 = nn.Linear(in_features=784, out_features=latent_dim)
        self.relu_1 = nn.ReLU()

    def forward(self, x):
        x = self.relu_1(self.linear_1(self.flatten_1(x)))
        return x


class EncoderFCMedium(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.flatten_1 = nn.Flatten(start_dim=1)
        self.linear_1 = nn.Linear(in_features=784, out_features=616)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=616, out_features=latent_dim)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        x = self.relu_1(self.linear_1(self.flatten_1(x)))
        x = self.relu_2(self.linear_2(x))
        return x


class EncoderFCLarge(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.flatten_1 = nn.Flatten(start_dim=1)
        self.linear_1 = nn.Linear(in_features=784, out_features=716)
        self.relu_1 = nn.ReLU()

        self.linear_2 = nn.Linear(in_features=716, out_features=684)
        self.relu_2 = nn.ReLU()

        self.linear_3 = nn.Linear(in_features=684, out_features=616)
        self.relu_3 = nn.ReLU()

        self.linear_4 = nn.Linear(in_features=616, out_features=latent_dim)
        self.relu_4 = nn.ReLU()

    def forward(self, x):
        x = self.relu_1(self.linear_1(self.flatten_1(x)))
        x = self.relu_2(self.linear_2(x))
        x = self.relu_3(self.linear_3(x))
        x = self.relu_4(self.linear_4(x))
        return x


class EncoderCNNSmall(nn.Module):

    # according to https://codeocean.com/capsule/9570390/tree/v1

    def __init__(self, dropout_prob=0.0):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.dropout_1 = nn.Dropout2d(dropout_prob)
        self.bn_1 = nn.BatchNorm2d(16)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_1 = nn.ReLU()  # 16x14x14

        self.conv_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.dropout_2 = nn.Dropout2d(dropout_prob)
        self.bn_2 = nn.BatchNorm2d(16)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.relu_2 = nn.ReLU()  # 16x8x8

        self.conv_3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.dropout_3 = nn.Dropout2d(dropout_prob)
        self.bn_3 = nn.BatchNorm2d(16)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_3 = nn.ReLU()  # 16x4x4

    def forward(self, x):
        x = self.relu_1(self.pool_1(self.bn_1(self.dropout_1(self.conv_1(x)))))
        x = self.relu_2(self.pool_2(self.bn_2(self.dropout_2(self.conv_2(x)))))
        x = self.relu_3(self.pool_3(self.bn_3(self.dropout_3(self.conv_3(x)))))

        return x


class EncoderCNNLarge(nn.Module):

    def __init__(self, dropout_prob=0.2):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.dropout_1 = nn.Dropout2d(dropout_prob)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu_1 = nn.ReLU()  # 64x32x32

        self.conv_2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
        self.dropout_2 = nn.Dropout2d(dropout_prob)
        self.bn_2 = nn.BatchNorm2d(64)
        self.relu_2 = nn.ReLU()  # 64x32x32

        self.conv_3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.dropout_3 = nn.Dropout2d(dropout_prob)
        self.bn_3 = nn.BatchNorm2d(128)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_3 = nn.ReLU()  # 128x16x16

        self.conv_4 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.dropout_4 = nn.Dropout2d(dropout_prob)
        self.bn_4 = nn.BatchNorm2d(128)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_4 = nn.ReLU()  # 128x8x8

        self.conv_5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.dropout_5 = nn.Dropout2d(dropout_prob)
        self.bn_5 = nn.BatchNorm2d(128)
        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu_5 = nn.ReLU()  # 128x4x4

    def forward(self, x):
        x = self.relu_1(self.bn_1(self.dropout_1(self.conv_1(x))))
        x = self.relu_2(self.bn_2(self.dropout_2(self.conv_2(x))))
        x = self.relu_3(self.pool_3(self.bn_3(self.dropout_3(self.conv_3(x)))))
        x = self.relu_4(self.pool_4(self.bn_4(self.dropout_4(self.conv_4(x)))))
        x = self.relu_5(self.pool_5(self.bn_5(self.dropout_5(self.conv_5(x)))))

        return x


class EncoderViT(nn.Module):

    def __init__(self, img_size, patch_size, embed_dim, num_heads, mlp_ratio, depth, in_chans=1, drop_path=0.0):
        super().__init__()

        norm_layer = nn.LayerNorm
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=drop_path)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.in_chans = in_chans

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

