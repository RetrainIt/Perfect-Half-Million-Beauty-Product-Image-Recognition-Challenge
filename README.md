# Perfect Half Million Beauty Product Image Recognition Challenge


This is my solution for [Perfect Half Million Beauty Product Image Recognition Challenge
](https://challenge2018.perfectcorp.com/index.html), which obtained the 2nd place (GAN) with MAP@7 0.29135.


### Requirement
- python 2.7
- pytorch 0.4.0
- PIL
- torchvision [here](https://github.com/Cadene/pretrained-models.pytorch)
- pretrainedmodels [here](https://github.com/facebookresearch/faiss)
- faiss [here](https://github.com/facebookresearch/faiss)
- download image data from [here](https://challenge2018.perfectcorp.com/index.html) if published
and place them in `./data/` , `./val/`  and `./test/`
- place `val.csv` and `test.csv` in `./`

### How to use
- run `python extract_all_feature.py`  
  extract feature at dir `./feature/`  
  - `-imsize `(default:480) : the size of longer image side used for extracting feature 
- run `python retrieval.py`  
  calcurate the MAP@7 between the data features and the val or test feature, the submission file at dir `./`  
  - `-iseval `(default: 1) : whether eval with the MAP@7
  - `-impath `(default: './data/') : the file for data
  - `-tpath `(default: './val/') : the file for val data, if the data is test data, change to './test/'
  -  `-tlabel `(default: './val.csv') : the label for val data, if the data is test data, please set `-iseval` as 0
 
Please cite the paper if you are using this code.

Zehang Lin, Zhenguo Yang, Feitao Huang and Junhong Chen. Regional Maximum Activations of Convolutions with Attention for Cross-domain Beauty and Personal Care Product Retrieval. ACM on multimedia conference, 2018.
