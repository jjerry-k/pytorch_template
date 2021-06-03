import os
import json
import logging

def save_dict_to_json(metrics, json_path, is_best=None):
    encoding = "w" if not os.path.exists(json_path) or is_best else "a"

    with open(json_path, encoding) as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        metrics = {key: float(value) for key, value in metrics.items()}
        json.dump(metrics, f, indent=4)

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