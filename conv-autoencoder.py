import torch
from torch import nn
from dpk import model_objs


class ConvAutoEncoder(torch.nn.Module):
    
    def __init__(self, bottleneck_chans=8):
        
        super(ConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=2, kernel_size=4, stride=4, padding=0, padding_mode="circular"),
                                    nn.Tanh(),
                                    nn.Conv2d(in_channels=2, out_channels=3, kernel_size=4, stride=3, padding=0, padding_mode="circular"),
                                    nn.Tanh(),
                                    nn.Conv2d(in_channels=3, out_channels=4, kernel_size=4, stride=3, padding=0, padding_mode="circular"),
                                    nn.Tanh(),
                                    nn.Conv2d(in_channels=4, out_channels=bottleneck_chans, kernel_size=4, stride=3, padding=0, padding_mode="circular"),
                                    nn.Tanh()
        )
                                    
        self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=bottleneck_chans, out_channels=8, kernel_size=4, stride=3, padding=2),
                                    nn.Tanh(),
                                    nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, stride=4, padding=2),
                                    nn.Tanh(),
                                    nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=4, stride=4, padding=2),
                                    nn.Tanh(),
                                    nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=4, stride=4, padding=2),
                                    nn.Tanh()
        )

    def forward(self, data):
        encoding = self.encoder(data)
        return self.decoder(encoding)

import xarray as xr

def get_data(path):
    data = xr.open_dataset(path)
    darr = data.to_array()[0, :, 0, :, :].to_numpy()  # remove elev and species dimensions
    
    darr = torch.tensor(darr)[:, None, :, :]  # shape (t, 1 chan, w, h)
    # darr = torch.transpose(torch.transpose(darr, dim0=0, dim1=2), dim0=0, dim1=1)[None, :, :, :]  # shape (1, w, h, t)
    
    return torch.log(darr + 1e-30)

    
from torch.optim import Adam

data = get_data(r"./data/NO2_" + "2018-01-31" + ".nc4") # 3d tensor of snapshots
data = data.cuda()
plt.imshow(data[0, 0, :, :], origin="lower")
# reproducibility
seed = 633

print("[ Using Seed : ", seed, " ]")

import torch
import random
import numpy as np
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

batch_size = 8
lr = 1e-5
wd = 1e-5

model = ConvAutoEncoder(bottleneck_chans=16)
model = model.cuda()
opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
epochs = 0
for i in range(500):
    for batch in range(data.shape[0] // batch_size):
        opt.zero_grad()
        batch_data = data[batch_size * batch:batch_size * (batch + 1)]
        reconstruction = model(batch_data)[:, :, 50:771, 100:1540]
        error = torch.sum((reconstruction - batch_data) ** 2)
        error.backward()
        opt.step()
    epochs += 1
model(batch_data).shape, batch_data.shape, model.encoder(batch_data)



# model = ConvAutoEncoder(bottleneck_chans=16)
# param_str = f"ConvAutoEncoder_{batch_size}_{epochs}_{seed}_{wd}_{lrt}"
# model.load_state_dict(torch.load(f"./forecasts/model_{param_str}.pt"))
# torch.save(model.state_dict(), f"./forecasts/model_{param_str}.pt")