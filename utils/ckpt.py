import os
import shutil
import torch

def save_checkpoint(state, is_best, checkpoint):
    filepath = os.path.join(checkpoint, f"epoch{state['epoch']:03d}.pth.tar")
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint, exist_ok=True)
    
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "best.pth.tar"))


def load_checkpoint(checkpoint, model, optimizer=None, map_location=None):
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return model, checkpoint, optimizer