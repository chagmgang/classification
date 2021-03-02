import torch
import torchvision

def make_dataset(is_train, batch_size):
    if is_train:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((33, 33)),
            torchvision.transforms.RandomCrop((32, 32)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(degrees=45),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(),
                ], p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
            ])

    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((33, 33)),
            torchvision.transforms.CenterCrop((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
            ])

    dataset = torchvision.datasets.CIFAR100(
            root='data', train=is_train, transform=transforms, download=True)
    dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=is_train, drop_last=is_train,
            batch_size=batch_size, num_workers=16, pin_memory=True)

    return dataloader

if __name__ == '__main__':
    dataloader = make_dataset(is_train=True, batch_size=32)
    dataloader = make_dataset(is_train=False, batch_size=32)
