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
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# GPU mode
# Please execute after installing cuda11
conda install -y -c pytorch cudatoolkit=11.0 # if using conda
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```


## Training Example 

### Classification
1. Download example dataset 
    - Flower: https://drive.google.com/file/d/1mDq2Oqwx_inTMUznA5KhiGtYOA3Vs_hb/view?usp=sharing
    - Cats & Dogs: https://drive.google.com/file/d/1uIWfyF8R6-WeumTSU4M0YuYw2X_QIfn8/view?usp=sharing
    - Dog Breed: https://drive.google.com/file/d/14FUyv7TzRq7T0r-ouwB9sdYOTPG22HIX/view?usp=sharing
2. Unzip dataset in `data` directory
3. Edit `config.yaml`
4. Execute command `python train.py --config config.yaml`

### Segmentation
...ing
1. Download example dataset

## To Do List
- [ ] Multi GPU
    - pretrained model 에 적용이 안되는 현상 발견. -> DP
- [ ] Inference code
- [ ] Segmentation