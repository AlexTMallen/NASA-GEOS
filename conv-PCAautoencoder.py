import time
import numpy as np
import random
from torch.optim import Adam
import torch
from torch import nn
import os


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class ConvPCAAutoEncoder(torch.nn.Module):

    def __init__(self, PCA_rank=20, bottleneck_chans=128, bottleneck_dim=1000, U_path=".\\data\\U20.pt"):

        super(ConvPCAAutoEncoder, self).__init__()
        self.U = torch.tensor(np.load(U_path), dtype=torch.float32)[
            None, :, :].cuda()  # (1, space, rank)

        self.encoder = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, dilation=1, padding=2, padding_mode="circular"),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5,
                                               dilation=1, padding=2, padding_mode="circular"),
                                     nn.MaxPool2d(kernel_size=4),
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                                               dilation=1, padding=2, padding_mode="circular"),
                                     nn.MaxPool2d(kernel_size=3),
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,
                                               dilation=1, padding=2, padding_mode="circular"),
                                     nn.MaxPool2d(kernel_size=3),
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channels=128, out_channels=bottleneck_chans, kernel_size=3,
                                               dilation=1, padding=1, padding_mode="circular"),
                                     nn.MaxPool2d(kernel_size=3),
                                     nn.ReLU(True)
                                     )
        # this must be changed if the above is modified
        self.enLinear = nn.Linear(
            in_features=3 * 6 * bottleneck_chans, out_features=bottleneck_dim)

        self.decoder1 = nn.Sequential(nn.Linear(in_features=bottleneck_dim, out_features=3 * 6 * bottleneck_chans),
                                      Reshape((-1, bottleneck_chans, 3, 6)),
                                      nn.ConvTranspose2d(
                                          in_channels=bottleneck_chans, out_channels=128, kernel_size=4, dilation=1, stride=2, padding=2),
                                      nn.ReLU(True),
                                      nn.ConvTranspose2d(
                                          in_channels=128, out_channels=128, kernel_size=4, dilation=1, stride=3, padding=2),
                                      nn.ReLU(True),
                                      nn.ConvTranspose2d(
                                          in_channels=128, out_channels=128, kernel_size=4, stride=3, padding=2),
                                      nn.ReLU(True),
                                      nn.ConvTranspose2d(
                                          in_channels=128, out_channels=64, kernel_size=4, dilation=1, stride=3, padding=2),
                                      nn.ReLU(True),
                                      nn.ConvTranspose2d(
                                          in_channels=64, out_channels=32, kernel_size=4, dilation=1, stride=3, padding=2),
                                      nn.ReLU(True),
                                      nn.Conv2d(
                                          in_channels=32, out_channels=32, kernel_size=3, dilation=1, stride=1, padding=1, padding_mode="circular"
        ),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                                          in_channels=32, out_channels=16, kernel_size=7, stride=5, padding=2),
            nn.ReLU(True))

        self.outputConv1 = nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=5, padding=2, padding_mode="circular")
        self.outputConv2 = nn.Conv2d(
            in_channels=8, out_channels=1, kernel_size=5, padding=2, padding_mode="circular")

    def forward(self, data):
        # data of shape (batch, 1, w, h)
        x = self.encoder(data)
        # shape (batch, 1, w*h)
        flattened = nn.Flatten(start_dim=2, end_dim=-1)(data)
        V = (flattened @ self.U)[:, 0, :]  # shape (batch, rank)
        print(x.detach().shape)
        x = nn.ReLU()(self.enLinear(nn.Flatten(start_dim=1, end_dim=-1)(x)))
        x = torch.cat(x, V)
        print(x.detach())
        x = self.decoder1(x)
        # print(x.detach().shape)

        x = x[:, :, 5:726, 10:1450] + self.map
        # / (self.outputConv1.kernel_size[0] * self.outputConv1.kernel_size[1])
        x = nn.ReLU()(self.outputConv1(x))
        x = self.outputConv2(x)
        # shape (batch, 1, space)?
        PCA_recon = V @ torch.transpose(self.U, dim0=1, dim1=2)
        x += PCA_recon.reshape((data.shape[0], 1, 721, 1440))
        # print(x.detach()[:, :, 100:-100, 100:-100].max())
        return x


def get_data(path):
    """ returns data of shape (t, 1, w, h) representing a series of log-images"""
    return torch.tensor(np.load(path))


# reproducibility
seed = 633

# print("[ Using Seed : ", seed, " ]")

# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True

batch_size = 2
lr = 1e-4
wd = 0

mean_over_time = torch.tensor(np.load("./data/map_mean_over_time.npy"))
data = get_data(os.path.join(r"./data", "NO2_all.npy")).type(torch.float32)
data -= mean_over_time
print(data.mean(), data.std(), data.std(axis=0).mean())
data = data.cuda()
model = ConvAutoEncoder(bottleneck_chans=128, bottleneck_dim=1000,
                        map_path="./data/map_normalized.npy")
print(model.parameters)
print([len(p) for p in model.parameters()])
model = model.cuda()
opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
epochs = 0

for i in range(100):
    print("\nEPOCH", epochs)
    # for fname in os.listdir(r"./data"):
    # if not fname.startswith("NO2_all"):
    #    continue
    epoch_mse = 0
    data = data[torch.randperm(data.shape[0])]
    for batch in range(data.shape[0] // batch_size):
        opt.zero_grad()
        batch_data = data[batch_size * batch:batch_size * (batch + 1)]
        reconstruction = model(batch_data)
        error = torch.mean((reconstruction - batch_data) ** 2)
        error.backward()
        print("data variance:", float(torch.mean(batch_data**2).cpu()))
        print("MSE:", float(error.detach().cpu()))
        epoch_mse += float(error.detach().cpu())
        opt.step()
    epochs += 1
    print("EPOCH MSE:", epoch_mse)

param_str = f"ConvAutoEncoder_{time.time()}_{batch_size}_{epochs}_{seed}_{wd}_{lr}"
torch.save(model.state_dict(), f"./model_{param_str}.pt")
torch.save(reconstruction, f"./reconstruction_{param_str}.pt")
