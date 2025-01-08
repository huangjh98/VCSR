# [Visual Contextual Semantic Reasoning for Cross-Modal Drone Image-Text Retrieval](https://ieeexplore.ieee.org/document/10634572)

##### Author: Jinghao Huang 
--------------------------
## Environment

Python 3.8.0  
Pytorch 1.11.0  
torchvision 0.12.0  
librosa 0.9.1  
numpy 1.21.6  
tqdm 4.64.0  

--------------------------
## Dataset
We use the following 2 datasets: ERA Image-Text Dataset and UDV Image-Text Dataset. [Dataset download link](https://pan.baidu.com/s/1ZVNe-H3GIJ4HHinqyKf3qQ?pwd=ul7f)

--------------------------
## Train

We train our model on a single 3090Ti GPU card. To train on different datasets, one needs to modify the configuration file in the code and then use the following training command:

 python train.py 

--------------------------
## Test

 python test.py
