import logging
from tqdm import tqdm
import torch

def eval(epoch, model, optimizer, criterion, dataloader, metrics, device, params):
    """
    Train 1 epoch 
    """
    model.eval()
    
    performance_dict = {
        "epoch": epoch+1
    }

    summ = {
        "val_loss": 0
    }

    summ.update({f"val_{metric}":0 for metric in metrics})

    # Training 1 Epoch
    with tqdm(total=len(dataloader)) as t:
        t.set_description(f'[{epoch+1}/{params["TRAIN"]["EPOCHS"]}]')
        
        # Iteration step
        with torch.no_grad():
            for i, (batch_image, batch_label) in enumerate(dataloader):
                
                batch_image, batch_label = batch_image.to(device), batch_label.type(torch.long).to(device)            
                predictions = model(batch_image)

                # Calculate Loss
                loss = criterion(predictions, batch_label)
                summ["val_loss"] += loss.item()

                # Calculate Metrics
                predictions = predictions.cpu().detach().numpy()
                batch_label = batch_label.cpu().detach().numpy()
                for metric, fn in metrics.items():
                    summ[f"val_{metric}"] += fn(predictions, batch_label)

                t.set_postfix({key: f"{val/(i+1):05.3f}"for key, val in summ.items()})
                t.update()
        
        performance_dict.update({key: val/(i+1) for key, val in summ.items()})
    return performance_dict