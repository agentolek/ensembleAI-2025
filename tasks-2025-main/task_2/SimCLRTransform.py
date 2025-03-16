import torchvision.transforms as transforms


mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]


class SimCLRTransform:
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=32),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        img1 = self.base_transform(img)
        img2 = self.base_transform(img)
        return img1, img2