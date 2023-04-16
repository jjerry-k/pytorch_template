# PyTorch Template

---

지극히 개인적인 일반적인 구조의 CNN 전용 template
- Classification, Segmentation 에 적합한(?) template

## Directory
- data
    - Dataset Name
        - train
        - validation
        - test

- log
    - Experiment name
        - date
            - ...

- model
    - loss.py
    - metric.py
    - model.py

- trainer
    - dataloader.py
    - evaluation.py
    - training.py

- utils
    - ckpt.py
    - log.py  


## Install Package
- Using Package
    - torch==1.7.1 
    - torchvision==0.8.2
    - torchaudio==0.7.2

``` bash
# Create environment
conda create -y -n torch python=3.7
conda activate torch

# CPU mode
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# GPU mode
pip install --no-cache-dir torch torchvision torchaudio
pip install -r requirements.txt
```


## Training Example 

### Classification
1. Download example dataset 
    - Flower: https://drive.google.com/file/d/1tGUTmiwltOelsGG2kby2r_sYPDRth-2T/view?usp=sharing
    - Cats & Dogs: https://drive.google.com/file/d/1uIWfyF8R6-WeumTSU4M0YuYw2X_QIfn8/view?usp=share_link
    - Dog Breed: https://drive.google.com/file/d/14FUyv7TzRq7T0r-ouwB9sdYOTPG22HIX/view?usp=sharing
2. Unzip dataset in `data` directory
3. Edit `config.yaml`
4. Execute command `python train.py --config config.yaml`

### Segmentation
...ing
1. Download example dataset

## To Do List
- [ ] Multi GPU
- [ ] Inference code
- [ ] Weights & Biases 연결?
- [ ] Segmentation