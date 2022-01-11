import os
import json
import logging

def save_dict_to_json(metrics, json_path, is_best=None):
    cond = False if (not os.path.exists(json_path)) else True
    metrics = {key: float(value) for key, value in metrics.items()}
    if cond:
        with open(json_path, "r") as tmp_f:
            data = json.load(tmp_f)
        data.update({metrics["epoch"]: metrics})
    else: 
        data = {metrics["epoch"]: metrics}
        
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

def set_logger(path, remove=True):
    
    if remove:
        if os.path.exists(path):
            os.remove(path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)