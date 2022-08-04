from model import Glow
import torch
from torchvision.utils import save_image
import os

test_dir = './test/'


device = torch.device("cpu")

if not os.path.exists(test_dir):
    os.makedirs(test_dir)


def calc_z_shapes(n_channel, input_size, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


z_sample = []
z_shapes = calc_z_shapes(n_channel=3, input_size=256, n_block=6)


glow = Glow(in_channel=3, n_flow=32, n_block=6)
glow =glow.to(device)
checkpoint = torch.load("./train/checkpoint/model_120001.pth", map_location=device)
glow.load_state_dict(checkpoint)

for i in range(10):

    for z in z_shapes:
            z_new = torch.randn(4, *z) * 0.7
            z_new = z_new.to(device)
            z_sample.append(z_new)


    

    img = glow.reverse(z_sample).cpu().data
    save_image(img, test_dir+f"./gen_{i+1}.png", nrow=2, normalize=True, range=(-0.5, 0.5))
    print(f"generate {i+1} image done!")
    i += 1
