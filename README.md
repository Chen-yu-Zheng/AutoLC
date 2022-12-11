# AutoLC
[ICPR 2022]  [AutoLC: Search Lightweight and Top-Performing Architecture for Remote Sensing Image Land-Cover Classification](https://ieeexplore.ieee.org/document/9956372). Including implementations of AutoDeepLab, Deeplab, Deeplabv3, Deeplabv3+, U-Net, FCN, PSPNet on LoveDA, Cityscapes and Floodnet data sets.

## File description

* mypath.py: path to each dataset (including loveda, cityscape, floodnet)
* autodeeplab.py: architecture Level  search space of AutoDeepLab
* cell_level_search.py: cell-level search space of AutoDeepLab
* genotypes.py: module candidates of cell-level search space
* modeling, operations.py: all basic models and operations included in our paper
* decode.py: decode searched architecture
* decoding_formulas.py: concrete implement for decoding architecture
* retrain_model: the model we proposed with lightweight encoder and decoder
* search.py: neural architecture search
* retrain.py: retrain the network after decoding
* test.py: get predictions on test set
* stats: get parameters and computational consumption of different models

## Training Procedure

**All together there are 3 stages:**

1. Architecture Search - Here you will train one large relaxed architecture that is meant to represent many discreet smaller architectures woven together. See search.py.

2. Decode - Once you've finished the architecture search, load your large relaxed architecture and decode it to find your optimal architecture. See decode.py: decode searched architecture.

3. Re-train - Once you have a decoded and poses a final description of your optimal model, use it to build and train your new optimal model. See retrain.py.

**Hardware Requirement**

* For architecture search with config in config_utils/search_args.py, you need at least an 12G GPU.


 ## Reproduce

**Search the lightweight encoder**

```shell
bash scripts/search_loveda.sh
```

**Decode to get the encoder**

```shell
bash scripts/decode.sh
```

**Retrain the decoded architecture with lightweight decoder**

```shell
bash scripts/retrain.sh
```

**Get prediction on the test set**

```shell
bash scripts/test.sh
```

## Requirements

* Pytorch
* Python 3
* tensorboardX
* torchvision
* tqdm
* numpy
* pandas
* apex

## Citation

If you use AutoLC in your research, please cite our ICPR2022 paper.

> ```latex
> @inproceedings{DBLP:conf/icpr/ZhengWMZ22,
>   author    = {Chenyu Zheng and
>                Junjue Wang and
>                Ailong Ma and
>                Yanfei Zhong},
>   title     = {AutoLC: Search Lightweight and Top-Performing Architecture for Remote
>                Sensing Image Land-Cover Classification},
>   booktitle = {26th International Conference on Pattern Recognition, {ICPR} 2022,
>                Montreal, QC, Canada, August 21-25, 2022},
>   pages     = {324--330},
>   publisher = {{IEEE}},
>   year      = {2022},
>   url       = {https://doi.org/10.1109/ICPR56361.2022.9956372},
>   doi       = {10.1109/ICPR56361.2022.9956372},
>   timestamp = {Thu, 01 Dec 2022 15:50:19 +0100},
>   biburl    = {https://dblp.org/rec/conf/icpr/ZhengWMZ22.bib},
>   bibsource = {dblp computer science bibliography, https://dblp.org}
> }
> ```

## References
[1] : [Thanks for NoamRosenbergs autodeeplab model implemention](https://github.com/NoamRosenberg/autodeeplab)
