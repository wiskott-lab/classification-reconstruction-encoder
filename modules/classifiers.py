import torch.nn as nn


class ClassifierFCSmall(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=latent_dim, out_features=10)

    def forward(self, x):
        x = self.linear_1(x)
        return x


class ClassifierFCMedium(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=latent_dim, out_features=12)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=12, out_features=10)

    def forward(self, x):
        x = self.relu_1(self.linear_1(x))
        x = self.linear_2(x)
        return x


class ClassifierFCLarge(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=latent_dim, out_features=64)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.relu_1(self.linear_1(x))
        x = self.linear_2(x)
        return x


class ClassifierCNNSmall(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten_2 = nn.Flatten(start_dim=1)
        self.linear_2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = self.linear_2(self.flatten_2(x))
        return x


class ClassifierCNNLarge(nn.Module):

    def __init__(self, dropout_prob=0.2):
        super().__init__()

        self.conv_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.dropout_1 = nn.Dropout2d(dropout_prob)
        self.bn_1 = nn.BatchNorm2d(256)
        self.relu_1 = nn.ReLU()  # 256x4x4

        self.conv_2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.dropout_2 = nn.Dropout2d(dropout_prob)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn_2 = nn.BatchNorm2d(256)
        self.relu_2 = nn.ReLU()  # 256x2x2

        self.flatten_3 = nn.Flatten(start_dim=1)
        self.linear_3 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        x = self.relu_1(self.bn_1(self.dropout_1(self.conv_1(x))))
        x = self.relu_2(self.bn_2(self.pool_2((self.dropout_2(self.conv_2(x))))))
        x = self.linear_3(self.flatten_3(x))
        return x


class ClassifierViT(nn.Module):  # adapted from https://github.com/facebookresearch/mae
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.bn_1 = nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6)
        self.linear_1 = nn.Linear(in_features=embed_dim, out_features=num_classes)

    def forward(self, x):
        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        x = self.linear_1(self.bn_1(x))
        return x


