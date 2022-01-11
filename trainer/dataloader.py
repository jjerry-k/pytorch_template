import random
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class BaseDataset(Dataset):
    """
    Thi is example code.
    Customize your self.
    """
    # def __init__(self, data_dir, transform):
    #     self.data_dir = data_dir
    #     self.filename = os.listdir(os.path.join(data_dir, 'image'))
    #     self.transform = transform

    # def __len__(self):
    #     # return size of dataset
    #     return len(self.filename)

    # def __getitem__(self, idx):

    #     image = Image.open(os.path.join(self.data_dir, "image", self.filename[idx]))
    #     if len(image.getbands()) > 3:
    #         image = image.convert("RGB")
    #     image = self.transform(image)

    #     label = Image.open(os.path.join(self.data_dir, "mask", self.filename[idx]).replace('jpg', 'png'))
    #     if len(label.getbands()) > 1:
    #         image = image.convert("L")
    #     label = self.transform(label)
    #     return image, label
    
    # For classification 

    def __init__(self, data_dir, transform):
        IMG_FORMAT = ["jpg", "jpeg", "bmp", "png", "tif", "tiff"]
        self.filelist = []
        self.classes = sorted(os.listdir(data_dir))
        for root, sub_dir, files in os.walk(data_dir):
            if not len(files): continue
            files = [os.path.join(root, file) for file in files if file.split(".")[-1] in IMG_FORMAT]
            self.filelist += files
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filelist)

    def __getitem__(self, idx):

        image = Image.open(self.filelist[idx])
        image = self.transform(image)
        label = self.filelist[idx].split('/')[-2]
        label = self.classes.index(label)
        return image, label

def Dataloader(config, transform=None):

    input_size = (config["INPUTSIZE"], config["INPUTSIZE"])

    dataloaders = {}

    for split in ['train', 'validation']:
        path = os.path.join(config["ROOT"], config["NAME"], split)
        transform_list = [transforms.Resize(input_size), transforms.ToTensor()]
        if split == 'train':
            transform_list = transform_list.insert(1, transform) if transform else transform_list
            dl = DataLoader(BaseDataset(path, transforms.Compose(transform_list)), batch_size=config["BATCHSIZE"], shuffle=True, num_workers=config["NUMWORKER"], drop_last=True)
        else:
            dl = DataLoader(BaseDataset(path, transforms.Compose(transform_list)), batch_size=config["BATCHSIZE"], shuffle=False, num_workers=config["NUMWORKER"])

        dataloaders[split] = dl

    return dataloaders