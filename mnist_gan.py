import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

def mnist_data():
    compose = transforms.Compose([transforms.ToTensor()])
    out_dir = '/data/'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU!")
else:
    device = torch.device("cpu")
    print("Running on CPU!")
