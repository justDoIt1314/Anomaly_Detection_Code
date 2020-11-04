# PyTorch implementation of "Three-branch Siamese Network for Anomaly Detection Without Labeled Data "

<p align="center"><img src="./MNAD_files/overview.png" alt="no_image" width="40%" height="40%" /><img src="./MNAD_files/teaser.png" alt="no_image" width="60%" height="60%" /></p>
This is the implementation of the paper "Three-branch Siamese Network for Anomaly Detection Without Labeled Data ".



## Dependencies
* Anaconda3-5.0.1-Windows
* Python 3.6
* PyTorch >= 1.0.0
* Numpy
* Sklearn

## Datasets
* USCD[[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* CUHK Avenue [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* Subway-entrance [[dataset](https://drive.google.com/drive/u/0/folders/0B8GCEsD4YSIkajlTSTB2aGxYNGs)]

Download the datasets into ``dataset`` folder, like ``./dataset/ped2/``

In addition, our parameters have been configured by default, and we can complete the evaluation and training only by running the following instructions.Our program has been tested in Win10+ CUDA10.0 + Pytorch1.2.0 environment.

## Training
* The training and testing codes are based on prediction method
```Windows PowerShell
git clone https://github.com/cvlab-yonsei/projects
To the current directory of the project
python Evaluate.py # for testing
```


## Pre-trained model and memory items
* Download our pre-trained model and memory items <br>Link: [[model and items](https://drive.google.com/file/d/11f65puuljkUa0Z4W0VtkF_2McphS02fq/view?usp=sharing)]
* Note that, these are from training with the Ped2 dataset

## Evaluation
* Test the model with our pre-trained model and memory items
```bash
python Evaluate.py --model_dir pretrained_model.pth --m_items_dir m_items.pt
```
* Test your own model
```Windows PowerShell
python Evaluate.py --model_dir your_model.pth --m_items_dir your_m_items.pt
```


