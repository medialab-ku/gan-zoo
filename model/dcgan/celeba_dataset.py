import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


class CelebAdataset(ImageFolder):
    def __init__(self,
                 root,
                 image_size):
        init_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        super().__init__(root, transform=init_transforms)
