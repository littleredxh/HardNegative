# Hard_Negative
This repo is for paper in ECCV2020: Hard negative examples are hard, but useful 

Arxiv link: https://arxiv.org/pdf/2007.12749.pdf

We provide the experiment result for each dataset(DATA) in folder exp_{DATA}.

DATA = CUB or CAR
The _result folder contains tensorboard results with Keys: '{DATA}_test_R_1', '{DATA}_test_R_2', '{DATA}_test_R_4', '{DATA}_test_R_8', '{DATA}_train_R_1', 'hn_ratio', 'loss'.

DATA = SOP
The _result folder contains tensorboard results with Keys: 'SOP_test_R_1', 'SOP_test_R_1', 'SOP_test_R_100', 'SOP_test_R_1000', 'SOP_train_R_1', 'hn_ratio', 'loss'.

DATA = ICR(Inshop)
The _result folder contains tensorboard results with Keys: 'ICR_test_R_1', 'ICR_test_R_1', 'ICR_test_R_100', 'ICR_test_R_1000', 'hn_ratio', 'loss'.


## Result Table

| CUB(ResNet50, embedding size 64)|
| Method |          R@1 |          R@2 |          R@4 |          R@8 | 
|     hn |     collapse |     collapse |     collapse |     collapse | 
|    shn | 56.72\pm0.65 | 68.64\pm0.33 | 78.60\pm0.24 | 86.64\pm0.26 | 
|    sct | 57.71\pm0.75 | 69.80\pm0.41 | 79.59\pm0.58 | 87.05\pm0.38 | 
   
| CAR(ResNet50, embedding size 64)|
| Method |          R@1 |          R@2 |          R@4 |          R@8 |
|     hn |     collapse |     collapse |     collapse |     collapse |
|    shn | 67.92\pm0.49 | 77.80\pm0.39 | 85.31\pm0.18 | 90.69\pm0.14 |
|    sct | 73.35\pm0.54 | 81.98\pm0.25 | 88.02\pm0.18 | 92.36\pm0.26 |
   
| SOP(ResNet50, embedding size 512)|
| Method |          R@1 |         R@10 |         R@20 |         R@40 |
|     hn | not measured | not measured | not measured | not measured |
|    shn | 81.06\pm0.06 | 92.32\pm0.07 | 96.84\pm0.05 | 98.90\pm0.01 |
|    sct | 81.90\pm0.07 | 92.61\pm0.06 | 96.77\pm0.04 | 98.75\pm0.02 |

| Inshop(ResNet50, embedding size 512)|
| Method |          R@1 |         R@10 |         R@20 |         R@40 |
|     hn | not measured | not measured | not measured | not measured |
|    shn | 90.55\pm0.15 | 97.37\pm0.07 | 98.09\pm0.10 | 98.45\pm0.08 |
|    sct | 90.93\pm0.22 | 97.51\pm0.05 | 98.16\pm0.03 | 98.44\pm0.03 |

| Hotel50K(ResNet50, embedding size 256)|
| Method |          R@1 |
|     hn |     collapse |
|    shn | 18.78\pm0.08 |
|    sct | 29.24\pm0.12 |
   
   
### HOTEL50K
unzip "exp_Hotel/input/dataset.zip" for chain retrieval

### Citation
```
@inproceedings{xuan2020hard,
  title={Hard negative examples are hard, but useful},
  author={Xuan, Hong and Stylianou, Abby and Liu, Xiaotong and Pless, Robert},
  booktitle={European Conference on Computer Vision},
  pages={126--142},
  year={2020},
  organization={Springer}
}
```

## Updates:
Feb 14, 2021: 
Improve loss function computation stablility
Improve the retrieval performance accross all datasets reported in the paper
Upload the training log information for all results
Add Hotel instance and chain retrieval code in "Hotel_instance_chain_retrieval.ipynb"
Add Hotel instance retrieval code（use in training） in  "_code.Evaluation.py"