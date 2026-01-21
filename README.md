# FSDENet
Fu jiahao

This is the code for our paper: FSDENet: A Frequency and Spatial Domains-Based Detail Enhancement Network for Remote Sensing Semantic Segmentation
https://ieeexplore.ieee.org/document/11051242
## Install

```shell
conda create -n airs python=3.8
conda activate airs
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
需要先将pip版本降低到24.1以下：
pip install pip==24.0 -i https://mirrors.aliyun.com/pypi/simple/
再安装：
pip install -r GeoSeg/requirements.txt
pip install pytorch_wavelets
```



## Data Preprocessing

Download the datasets from the official website and split them yourself.

  - [ISPRS Vaihingen and Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) 
  - [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)
  - Please follow the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isaid) to preprocess the iSAID dataset. 


Prepare the following folders to organize this repo:

```none
FGHFN
├── GeoSeg (代码)
├── fig_results (save the masks predicted by models)
├── test_log （测试日志）
├── weights (权重目录)
├── data (数据目录)
│   ├── loveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   ├── vaihingen
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
│   ├── potsdam 
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
```





```shell
cd FSDENet
```

**Potsdam**

```shell
python GeoSeg/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/train_images" \
--mask-dir "data/potsdam/train_masks" \
--output-img-dir "data/potsdam/train/images_1024" \
--output-mask-dir "data/potsdam/train/masks_1024" \
--mode "train" --split-size 1024 --stride 1024 --rgb-image 
```

```shell
python GeoSeg/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks_eroded" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded --rgb-image
```

```shell
python GeoSeg/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt --rgb-image
```


**LoveDA**

```shell
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/loveDA/Train/Rural/masks_png --output-mask-dir data/loveDA/Train/Rural/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/loveDA/Train/Urban/masks_png --output-mask-dir data/loveDA/Train/Urban/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/loveDA/Val/Rural/masks_png --output-mask-dir data/loveDA/Val/Rural/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/loveDA/Val/Urban/masks_png --output-mask-dir data/loveDA/Val/Urban/masks_png_convert
```

## Training

"-c" means the path of the config, use different **config** to train different models.

```shell
cd FSDENet
```

**Potsdam**

```shell
python GeoSeg/train_supervision.py -c GeoSeg/config/potsdam/FSDENet_potsdam.py 
```

**Vaihingen**

```shell
python GeoSeg/train_supervision_dp.py -c GeoSeg/config/vaihingen/FSDENet_vaihingen.py
```

**LoveDA** 

```shell
python GeoSeg/train_supervision_dp.py -c GeoSeg/config/loveda/FSDENet_loveda.py
```

**iSAID**

```shell
python train_supervision.py -c GeoSeg/config/loveda/FSDENet_isaid.py
```





## Testing

```shell
cd FSDENet
```


**Potsdam**

```shell
python GeoSeg/test_potsdam.py -c GeoSeg/config/potsdam/FSDENet_potsdam.py -o ~/fig_results/potsdam/FGHFN_potsdam --rgb -t 'd4'
```

**Vaihingen**

```shell
python GeoSeg/test_vaihingen.py -c GeoSeg/config/vaihingen/FSDENet_vaihingen.py -o ~/fig_results/FGHFN_vaihingen/ --rgb -t "d4"
```

**LoveDA** 

```shell
python GeoSeg/test_loveda.py -c GeoSeg/config/loveda/FSDENet_loveda.py -o ~/fig_results/loveda/FGHFN_loveda --rgb --val -t "d4"
```

**iSAID**

```shell
python GeoSeg/test_isaid.py -c GeoSeg/config/isaid/FSDENet_isaid.py -o ~/fig_results/isaid/convlsrnet_isaid/  -t "d4"
```



## Acknowledgement

Our training scripts comes from [GeoSeg](https://github.com/WangLibo1995/GeoSeg). Thanks for the author's open-sourcing code.

- [GeoSeg(UNetFormer)](https://github.com/WangLibo1995/GeoSeg)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
