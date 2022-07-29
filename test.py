from model import Glow
import torch
from torchvision.utils import save_image

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
z_shapes = calc_z_shapes(n_channel=3, input_size=64, n_block=4)


for i in range(20):

    for z in z_shapes:
            z_new = torch.randn(64, *z) * 0.7
            z_new = z_new
            z_sample.append(z_new)


    glow = Glow(in_channel=3, n_flow=32, n_block=4)
    checkpoint = torch.load("./model_190001.pth")
    glow.load_state_dict(checkpoint)

    img = glow.reverse(z_sample).cpu().data
    save_image(img, f"./gen_{i+1}.png", nrow=8, normalize=True, range=(-0.5, 0.5))
    i += 1






