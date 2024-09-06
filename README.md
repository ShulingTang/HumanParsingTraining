# Human Parsing Training

This repository provides detailed instructions on how to train a human parsing model and currently supports the following architectures:

1. [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)
2. [CDGNet](https://github.com/tjpulkl/CDGNet)
3. [SOLIDER-HumanParsing](https://github.com/tinyvision/SOLIDER-HumanParsing/tree/master)
4. SWIN-CDG(all of 1, 2, 3)

It offers single-GPU training, single-machine multi-GPU training, and multi-machine multi-GPU training (which will be updated later).

Our code is modified from [CDGNet](https://github.com/tjpulkl/CDGNet) and [SOLIDER-HumanParsing](https://github.com/tinyvision/SOLIDER-HumanParsing/tree/master), 
and you can obtain more information from the original repositories.


## Installation and Datasets

Details of installation and dataset preparation can be found in [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing).

### Docker Train Environment

```shell
docker pull pixocial.azurecr.io/train/humanparsing:v1.0
```


## Prepare Pre-trained Models

Step 1. Download models from [SOLIDER](https://github.com/tinyvision/SOLIDER), or use [SOLIDER](https://github.com/tinyvision/SOLIDER) to train your own models.
（pretrained model resnet101 download from [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)）

Steo 2. Put the pretrained models under the `pretrained` file, and rename their names as `./pretrained/solider_swin_tiny(small/base).pth`

## Training
Train with single GPU or multiple GPUs:

```shell
sh train_swin.sh
```

## Performance

The following metrics are based on training with a dataset of 110,000 non-public data samples. 
You can try training on other open-source datasets, but we do not provide metrics for those here.

| Method  | pretraind model | OurData(MIoU) |
|---------|:---------------:|:-------------:| 
| ACE2P   |    resnet101    |     79.73     | 
| SELP    |   resnet101   |     78.45     | 
| CDGNet  |   resnet101   |     80.65     | 
| SOLIDER |    Swin Base    |     81.08     | 
| SWIN-CDG |    Swin Base    |       -       | 

- We use the pretrained models from [SOLIDER](https://github.com/tinyvision/SOLIDER) and [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)..
- The semantic weight we used in these experiments is 0.8.

## Citation

Here are the papers cited by this part of the code. If you find our code useful, you might consider citing them.

```
@inproceedings{chen2023beyond,
    title={Beyond Appearance: a Semantic Controllable Self-Supervised Learning Framework for Human-Centric Visual Tasks},
    author={Weihua Chen and Xianzhe Xu and Jian Jia and Hao Luo and Yaohua Wang and Fan Wang and Rong Jin and Xiuyu Sun},
    booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2023},
}
  
@InProceedings{Liu_2022_CVPR,
    author    = {Liu, Kunliang and Choi, Ouk and Wang, Jianming and Hwang, Wonjun},
    title     = {CDGNet: Class Distribution Guided Network for Human Parsing},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {4473-4482}
}

@article{li2020self,
    title={Self-Correction for Human Parsing}, 
    author={Li, Peike and Xu, Yunqiu and Wei, Yunchao and Yang, Yi},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    year={2020},
    doi={10.1109/TPAMI.2020.3048039}}
```

