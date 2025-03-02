import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from model.unet import UNet
from model.unetconditional import UNetConditional
import numpy as np
from argparse import ArgumentParser

def arg_parser(parser):

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--c",  dest='cond', action="store_true", help="conditional generation")
    args, unknow = parser.parse_known_args()
    if unknow:
        msg = " ".join(unknow)
        print('[WARNING] [{:s}] Unrecognized arguments: {:s}'.format('run',msg) )
    return args

def main():
    if (torch.cuda.is_available()):
        print("found cuda device "+torch.cuda.get_device_name(0))
        device = torch.device("cuda:0")
    else:
        print("cannot find cuda device")

    if args.cond:
        print("conditional")
        model = UNetConditional(in_channels=1, out_channels=1)
        model.load_state_dict(torch.load("unetconditional_checkpoint.pth", weights_only=True))
    else:
       model = UNet(in_channels=1, out_channels=1)
       model.load_state_dict(torch.load("unet_checkpoint.pth", weights_only=True))
    
    model.to(device)

    col = 10
    # Gaussian noise input
    with torch.no_grad():
        fig, axes = plt.subplots(10, col, figsize=(15, 5))
        for i in range(col):
            for num in range(10):
                n=50
                noise_img = torch.randn(1, 1, 64, 64).to(device) 
                x = noise_img
                for j in range(n):
                    mixfactor=1/(n-j)
                    if args.cond:
                        output = model(x,class_label=torch.unsqueeze(torch.tensor(num).to(device),0))
                    else:
                        output = model(x)
                    x = x*(1-mixfactor) + output*mixfactor
                
                #axes[num, i].imshow(noise_img.cpu().squeeze(0).squeeze(0), cmap='gray')
                #axes[num, i].axis('off')
                axes[num, i].imshow(x.cpu().squeeze(0).squeeze(0), cmap='gray')
                axes[num, i].axis('off')
        plt.show()
        if args.cond:
            fig.savefig('val_result-cond.png')
        else:
            fig.savefig('val_result.png')

if __name__ == "__main__":
    parser = ArgumentParser()
    args   = arg_parser(parser)
    main()