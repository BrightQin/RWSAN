# RWSAN
This repo is the official implementation of "**Deep Residual Weight-Sharing Attention Network with Low-Rank Attention for Visual Question Answering**" published in the **IEEE Transactions on Multimedia (TMM), 2022**.

## Citation
If this repository is helpful for your research, we'd really appreciate it if you could cite the following paper:

```
@ARTICLE{9770348,
  author={Qin, Bosheng and Hu, Haoji and Zhuang, Yueting},
  journal={IEEE Transactions on Multimedia}, 
  title={Deep Residual Weight-Sharing Attention Network with Low-Rank Attention for Visual Question Answering}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2022.3173131}}
  ```

## Hardware Requirements
1. At least one 1 Nvidia GPU.
2. 50GB free disk space.

## Software Requirements
1. Install [Python](https://www.python.org/downloads/) >= 3.6 (Project works with Python 3.6~3.9)
2. Install [Cuda](https://developer.nvidia.com/cuda-toolkit) >= 10.1 and [cuDNN](https://developer.nvidia.com/cudnn) (Project works with cudatookit 10.1, 10.2, and 11.1)
3. Install [PyTorch](http://pytorch.org/) >= 1.7.0 with CUDA (Project works with PyTorch 1.7.0~1.9.1)

## Dataset
Please download the processed VQA-v2 and VG datasets in [OneDrive](https://zjueducn-my.sharepoint.com/:f:/g/personal/3170105600_zju_edu_cn/EqXAXyjnYE1Dn4hMOoRnO6IBV78-cS2HSJsW2vZzpmKkaQ?e=cHf8H9) or [BaiduYun](https://pan.baidu.com/s/19PdZwXWx2vhByfKxZt9oCw?pwd=rwsa) and place them in the root folder as follows:

```angular2html
|-- datasets
	|-- coco_extract
	|  |-- train2014.tar.gz
	|  |-- val2014.tar.gz
	|  |-- test2015.tar.gz
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


## Training and Evaluate on the *Val* Split
Run the following command to train the RWSAN on *train* split of VQA-v2 dataset and evaluate on the *val* split.

```bash
python3 run.py --RUN='train' --VERSION='RWSAN' --GPU='0' --SPLIT='train' --ACCU=1 --NW=4
```

### Command
1. ```--VERSION='RWSAN'``` the name of this experiment.
2. ```--GPU=str``` use the specific GPU device.
3. ```--SPLIT={'train', 'train+val', 'train+val+vg'}``` the training set you want to use.
4. ```--ACCU=1``` gradient accumulation for low memory GPU. ```1``` for not using gradient accumulation. Note that `BATCH_SIZE` must be divided by ```ACCU```. (The default `BATCH_SIZE` is 64, so the ```--ACCU``` can be 1, 2, 4, 8 ...).

The checkpoints are stored in ```./ckpts/ckpt_RWSAN/```, and the log files for average training loss and performace on *val* split in every epoch are stored in ```./results/log/'''

## Training and Evaluate on the *Test* Split
Offline evaluation support evaluation on the the VQA-v2 *val* split.

```bash
$ python3 run.py --RUN='val' --CKPT_V='ckpt_RWSAN' --CKPT_E=16
```

You could evaluate the model on *test-dev* and *test-std* splits of VQA-v2 dataset online.

```bash
$ python3 run.py --RUN='test' --CKPT_V='ckpt_RWSAN' --CKPT_E=16
```

Result files are stored in ```results/result_test/result_run_<'PATH+random number' or 'VERSION+EPOCH'>.json```

Upload the result file to [Eval AI](https://eval.ai/web/challenges/challenge-page/830/overview) to see the scores on *test-dev* and *test-std* splits.

## Thanks
[MCAN](https://github.com/MILVLG/mcan-vqa)

[bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)
