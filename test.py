# %%
import os
import yaml
import shutil
import logging
import datetime
import argparse

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pprint import pprint

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

import model
import trainer
import utils

# %%
ROOT = './log/flower_photos/2021-06-24-00-59-48'
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
config = utils.config_parser(CONFIG_PATH)
classes = sorted(os.listdir(os.path.join("./data/", config["DATA"]["NAME"], "train")))
# %%
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= ",".join(str(gpu) for gpu in config["COMMON"]["GPUS"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Build Model")
net = model.Model(config["MODEL"]["BASEMODEL"], config["MODEL"]["NUMCLASSES"], config["MODEL"]["FREEZE"]).to(device)
print("Load Checkpoint")
ckpt_path = os.path.join(ROOT, "best.pth.tar")
net, _, _ = utils.load_checkpoint(ckpt_path, net)
# %%
selected_class = classes[np.random.randint(0, len(classes))]
test_img_root = f'./data/flower_photos/validation/{selected_class}'
test_img_list = os.listdir(test_img_root)
selected_image = test_img_list[np.random.randint(0, len(test_img_list))]
test_img_path = os.path.join(test_img_root, selected_image)
raw_image = Image.open(test_img_path)
input_size = (config["DATA"]["INPUTSIZE"], config["DATA"]["INPUTSIZE"])
transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])
image = transform(raw_image).unsqueeze(0)

with torch.no_grad():
    pred = torch.softmax(net(image), dim=1)

plt.title(f"Label: {selected_class}, Predict: {classes[torch.argmax(pred).item()]}")
plt.imshow(raw_image)
plt.show()