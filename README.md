# RWSAN
This repo is the official implementation of "**Deep Residual Weight-Sharing Attention Network with Low-Rank Attention for Visual Question Answering**" published in the **IEEE Transactions on Multimedia (TMM), 2022**.

## Citation
If this repository is helpful for your research, we'd really appreciate it if you could cite the following paper:

## Hardware Requirements
1. At least one 1 Nvidia GPU (>= 8GB), 
2. 50GB free disk space.

## Software Requirements
1. Install [Python](https://www.python.org/downloads/) >= 3.6 (Project works with Python 3.6~3.9)
2. Install [Cuda](https://developer.nvidia.com/cuda-toolkit) >= 10.1 and [cuDNN](https://developer.nvidia.com/cudnn) (Project works with cudatookit 10.1, 10.2, and 11.1)
3. Install [PyTorch](http://pytorch.org/) >= 1.7.0 with CUDA (Project works with PyTorch 1.7.0~1.9.1)

## Dataset
The image features are extracted using the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) strategy. We release the extracted features in [OneDrive](https://zjueducn-my.sharepoint.com/:f:/g/personal/3170105600_zju_edu_cn/ErSY69_1A_dBgvRNQAIgf_UB4GJ9ajaPBr42jNtDTGfeqQ?e=dLhEZ1) or [BaiduYun](https://pan.baidu.com/s/1R9hMzK3KRplhRFj9_vKflQ?pwd=RWSA) (code: rwsa). Please place the dataset as follows:

```angular2html
|-- datasets
	|-- coco_extract
	|  |-- train2014.tar.gz
	|  |-- val2014.tar.gz
	|  |-- test2015.tar.gz
```

Besides, we use the VQA samples from the [visual genome dataset](http://visualgenome.org/) to expand the training samples. We provide processed vg questions and annotations files. Please download them from [OneDrive]() or [BaiduYun](), and place them as follow:

```angular2html
|-- datasets
	|-- vqa
	|  |-- VG_questions.json
	|  |-- VG_annotations.json
```

After that, run the following script to download QA files and unzip the images features.

```bash
$ sh setup.sh
```

Finally, the datasets folders shall have the following structure:

```angular2html
|-- datasets
	|-- coco_extract
	|  |-- train2014
	|  |  |-- COCO_train2014_...jpg.npz
	|  |  |-- ...
	|  |-- val2014
	|  |  |-- COCO_val2014_...jpg.npz
	|  |  |-- ...
	|  |-- test2015
	|  |  |-- COCO_test2015_...jpg.npz
	|  |  |-- ...
	|-- vqa
	|  |-- v2_OpenEnded_mscoco_train2014_questions.json
	|  |-- v2_OpenEnded_mscoco_val2014_questions.json
	|  |-- v2_OpenEnded_mscoco_test2015_questions.json
	|  |-- v2_OpenEnded_mscoco_test-dev2015_questions.json
	|  |-- v2_mscoco_train2014_annotations.json
	|  |-- v2_mscoco_val2014_annotations.json
	|  |-- VG_questions.json
	|  |-- VG_annotations.json

```


## Training 
Coming soon...

## Testing
Coming soon...
