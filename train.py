import os
import yaml
import shutil
import logging
import datetime
import argparse

import numpy as np
from pprint import pprint

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

import model
import trainer
import utils

optm_dict = {
    "adadelta": optim.Adadelta,
    "adam": optim.Adam,
    "asgd": optim.ASGD,
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "adamax": optim.Adamax,
    "rmsprop": optim.RMSprop,
    "rprop": optim.Rprop,
    "sgd": optim.SGD,
    "sparseadam": optim.SparseAdam
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml',
                    help="Path of configuration file")    

def main(config):
    # SET DEVICE
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ",".join(str(gpu) for gpu in config["COMMON"]["GPUS"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    curr_date = datetime.datetime.now()
    year, month, day = curr_date.year, curr_date.month, curr_date.day
    hour, minute, second = curr_date.hour, curr_date.minute, curr_date.second
    DATE = f"{year:04d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}"
    
    SAVEPATH = os.path.join("./log", config["DATA"]["NAME"], DATE)
    os.makedirs(SAVEPATH)
    utils.set_logger(os.path.join(SAVEPATH, "train.log"))
    write_yaml(os.path.join(SAVEPATH, "config.yaml"), config)

    # DATA LOADING
    logging.info(f'Loading {config["DATA"]["NAME"]} datasets')
    transform = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]
    loader = trainer.Dataloader(config["DATA"])

    # MODEL BUILD
    logging.info(f"Building model")
    net = model.Model(config["MODEL"]["BASEMODEL"], config["MODEL"]["NUMCLASSES"], config["MODEL"]["FREEZE"]).to(device)
    # net = model.Model(num_classes=config["MODEL"]["NUMCLASSES"]).to(device)
    
    if torch.cuda.is_available() and len(config["COMMON"]["GPUS"]) > 1:
        logging.info(f"Multi GPU mode")
        net = torch.nn.DataParallel(net, device_ids=config["COMMON"]["GPUS"]).to(device)

    criterion = model.loss_fn
    metrics = {"acc": model.accuracy} # If classification
    # metrics = {}
    optm = optm_dict[config["TRAIN"]["OPTIMIZER"]](net.parameters(), lr=config["TRAIN"]["LEARNINGRATE"])
    
    # TRAINING
    EPOCHS = config["TRAIN"]["EPOCHS"]
    logging.info(f"Training start !")
    best_val_loss = np.inf
    for epoch in range(EPOCHS):

        metrics_summary = trainer.train(epoch, net, optm, criterion, loader["train"], metrics, device, config)
        metrics_summary.update(trainer.eval(epoch, net, optm, criterion, loader["validation"], metrics, device, config))

        metrics_string = " ; ".join(f"{key}: {value:05.3f}" for key, value in metrics_summary.items())
        logging.info(f"[{epoch+1}/{EPOCHS}] Performance: {metrics_string}")

        is_best = metrics_summary['loss_val'] <= best_val_loss

        utils.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': net.state_dict(),
                                'optim_dict': optm.state_dict()},
                                is_best=is_best,
                                checkpoint=SAVEPATH)
        
        if is_best:
            logging.info("Found new best loss !")
            best_val_loss = metrics_summary['loss_val']

            best_json_path = os.path.join(
                SAVEPATH, "metrics_best.json")
            utils.save_dict_to_json(metrics_summary, best_json_path, is_best)

        last_json_path = os.path.join(
            SAVEPATH, "metrics_history.json")
        utils.save_dict_to_json(metrics_summary, last_json_path)

        # TODO: EARLY STOP
    logging.info(f"Training done !")


if __name__ == "__main__":
    
    # Load config file
    args = parser.parse_args()
    config = config_parser(args.config)

    # Execute main function
    main(config)