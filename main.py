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

def add_gaussian_noise(img, ratio):
    noise = torch.randn_like(img)
    return 0.9*img + ratio*noise

def sparsity_penalty(module, lambda_s=1e-4):
    penalty = 0.0
    for name, param in module.named_parameters():
        if 'weight' in name:  # Apply only to weight parameters
            penalty += torch.norm(param, p=2) ** 2  # L2 norm squared
    return lambda_s * penalty

def xavier_init(m):
    if isinstance(m, nn.Linear):
        # Xavier initialization for Linear layers
        nn.init.xavier_uniform_(m.weight)  # Or init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def main():
    # Set device
    if (torch.cuda.is_available()):
        print("found cuda device "+torch.cuda.get_device_name(0))
        device = torch.device("cuda:0")
    else:
        print("cannot find cuda device")
        return
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((64, 64)),
        transforms.RandomRotation((-30,30))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Initialize model
    if args.cond:
        print("conditional")
        model = UNetConditional(in_channels=1, out_channels=1).to(device)
    else:
        model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.apply(xavier_init)
    
    # Training loop
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            t_ = torch.tensor([[(5*epoch)+1]]).to(device)
            noisy_images = add_gaussian_noise(images,1+epoch/epochs).clamp(0, 1).to(device)
            
            optimizer.zero_grad()
            if args.cond:
                #penalty = sparsity_penalty(model.emb_to_bottleneck)
                outputs = model(noisy_images,labels,t_)
                loss = criterion(outputs, images)
            else:
                outputs = model(noisy_images)
                loss = criterion(outputs, images)
            
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.7f}")
    
    if args.cond:
        torch.save(model.state_dict(), "unetconditional_checkpoint.pth")
    else:
        torch.save(model.state_dict(), "unet_checkpoint.pth")
    
if __name__ == "__main__":
    parser = ArgumentParser()
    args   = arg_parser(parser)
    main()
