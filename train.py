from tqdm import tqdm
import argparse
from math import log
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Glow


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--dataset_root", default="./data/", type=str, help="the root path of training data")
parser.add_argument("--batch_size", default=16, type=int, help="batch size")
parser.add_argument("--img_size", default=64, type=int, help="image size")

parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument("--n_flow", default=32, type=int, help="number of flows in each block")
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument("--use_lu", default=True, type=bool, help="use LU decomposed invertible conv")
parser.add_argument("--affine", default=True, type=bool, help="use affine coupling instead of additive")
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")

parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")


checkpoint_dir = './train/checkpoint/'
sample_dir = './train/sample/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)






def calc_z_shapes(n_channel, input_size, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):

    n_pixel = image_size * image_size * 3

    c = -log(n_bins) * n_pixel
    loss = c + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )



def train(args, model, optimizer):
    dataset = iter(sample_data(args.dataset_root, args.batch_size, args.img_size))
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image, _ = next(dataset)
            image = image.to(device)

            image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

                image = image / n_bins - 0.5

            else:
                image = (image/ (n_bins-1)) - 0.5

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model(
                        image + torch.rand_like(image) / n_bins
                    )

                    continue

            else:
                log_p, logdet, _ = model(
                    image + torch.rand_like(image) / n_bins
                    )


            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model.zero_grad()
            loss.backward()
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )

            if i % 100 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model.reverse(z_sample).cpu().data,
                        sample_dir+f"{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

            if i % 10000 == 0:
                torch.save(
                    model.state_dict(), checkpoint_dir+f"model_{str(i + 1).zfill(6)}.pth"
                )
                torch.save(
                    optimizer.state_dict(), checkpoint_dir+f"optim_{str(i + 1).zfill(6)}.pth"
                )


if __name__ == "__main__":
    args = parser.parse_args()

    model = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=args.use_lu)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer)


