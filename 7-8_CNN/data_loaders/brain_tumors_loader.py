import torch
import torchvision
from torchvision.transforms import v2

def get_loaders(dataset_path='datasets/brain_tumors', batch_size=32):
    transform = v2.Compose([
        v2.Resize((128, 128)),
        v2.RandomCrop((128, 128)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = torchvision.datasets.ImageFolder(root=str(dataset_path + '/train'), transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = torchvision.datasets.ImageFolder(root=str(dataset_path + '/test'), transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    classes = trainset.class_to_idx
    
    return trainloader, testloader, classes