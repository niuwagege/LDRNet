## Requirements
* tensorflow==2.3.1
* tensorflow-addons==0.11.2
* opencv-python==4.4.0.46
* numpy==1.18.5

## Predict
```
python3 predict.py
```

## Train
```
CUDA_VSIBLE_DEVICES=0 python3 train.py --config_file ./config.yaml
```
###fill the label_path and img_folder_path in config.yaml
###label consists of 18 elements: file name, eights floats for four coordinates, eights float for weights(set to 0.0625), class index(start from 0)
