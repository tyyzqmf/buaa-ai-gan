from torch import nn


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
    # net.apply(init_weights)


class GenerateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = nn.Sequential(
            # input is Z, [B, 100, 1, 1] -> [B, 64 * 4, 4, 4]
            nn.ConvTranspose2d(in_channels=100, out_channels=64 * 4,
                               kernel_size=(4, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. [B, 64 * 4, 4, 4] -> [B, 64 * 2, 8, 8]
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. [B, 64 * 2, 8, 8] -> [B, 64, 16, 16]
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. [B, 64, 16, 16] -> [B, 1, 32, 32]
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.G(x)
        return x


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(
            # input [B, 1, 32, 32] -> [B, 64, 16, 16]
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # state size. [B, 64, 16, 16] -> [B, 128, 8, 8]
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2),
            # state size. [B, 128, 8, 8] -> [B, 256, 4, 4]
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2),
            # state size. [B, 256, 4, 4] -> [B, 1, 1, 1]
            nn.Conv2d(64 * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        x = self.D(x)
        return x


class MatchingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.M = nn.Sequential(
            # input [B, 1, 32, 32] -> [B, 64, 16, 16]
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # state size. [B, 64, 16, 16] -> [B, 128, 8, 8]
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2),
            # state size. [B, 128, 8, 8] -> [B, 256, 4, 4]
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2),
            # state size. [B, 256, 4, 4] -> [B, 1, 1, 1]
            nn.Conv2d(64 * 4, 128, 4, 1, 0, bias=False),
            nn.Conv2d(128, 100, 1, 1, 0))

    def forward(self, x):
        x = self.M(x)
        return x
