import numpy as np
import torch
import torchvision

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor()
])


class MNISTReader(torch.utils.data.Dataset):
    def __init__(self, noise=False, root='~/Datasets/MNIST'):
        super(MNISTReader, self).__init__()
        self.noise = noise
        self.train_dataset = torchvision.datasets.MNIST(
            root=root, train=True, download=True, transform=transform)

    def __getitem__(self, index):
        img, label = self.train_dataset[index]

        if self.noise:
            noise = np.random.randn(100, 1, 1).astype("float32")
            return img, noise
        else:
            return img

    def __len__(self):
        return len(self.train_dataset)


if __name__ == '__main__':
    for _sample in MNISTReader():
        print(_sample.shape)
        break
