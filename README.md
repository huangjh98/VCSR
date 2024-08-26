# Visual Contextual Semantic Reasoning for Cross-Modal Drone Image-Text Retrieval

##### Author: Jinghao Huang 

### -------------------------------------------------------------------------------------
### Welcome :+1:_<big>`Fork and Star`</big>_:+1:, then we'll let you know when we update

### -------------------------------------------------------------------------------------

##
## [VCSR](https://ieeexplore.ieee.org/document/10634572)

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
We use the following 2 datasets: ERA Image-Text Dataset and UDV Image-Text Dataset. [Dataset download link](VCSR/README.md)

--------------------------
## Train

We train our model on a single 3090Ti GPU card. To train on different datasets, one needs to modify the configuration file in the code and then use the following training command:

 python train.py 

--------------------------
## Test

 python test.py
