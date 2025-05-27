# AtSpeed
This is the pytorch implementation of our paper:
> [Efficient Inference for Large Language Model-based Generative Recommendation](https://arxiv.org/pdf/2410.05165) (ICLR 2025)

We also release a Python package, [BeamSD](https://github.com/transcend-0/BeamSD), which can accelerate the beam search generation of transformers by 1.5x speedup with just one line of code!

## Environment
- Anaconda 3
- Python 3.9.0
- pytorch 1.13.0
- transformers 4.41.0

## Usage
### Data

```bash
data/
├── beauty
├── games
```

The data in the floder is already processed and can be used directly. The raw data is from [Amazon product data](https://jmcauley.ucsd.edu/data/amazon/). 
We sort users' historical interactions by the global timestamps, and then split them into training, validation, and testing sets with the ratio of 8:1:1. If you want to apply this splitting method to your own dataset, please refer to the example for Beauty dataset in `data/data_process.ipynb`. 
For the item identifier, we follow [LC-Rec](https://github.com/RUCAIBox/LC-Rec) to set the length L = 4, *i.e.,* the token sequence length of a generated item would be 4.


### Train

#### Target Model

First, replace the parameters in `code/script/finetune_llama.sh` with your own parameters, such as `LOG_DIR`, `OUTPUT_DIR`, etc.

```bash
LOG_DIR=YOUR_LOG_DIR
OUTPUT_DIR=YOUR_OUTPUT_DIR
BASE_MODEL=YOUR_BASE_MODEL_PATH
```

Then, run the following command to train the target model.

```bash
cd code
bash script/finetune_llama.sh
```

#### Draft Model

1. Generate Teacher Data

Replace the parameters in `code/script/generate_teacher_data.sh` with your own parameters, and then run the following command.

```bash
cd code
bash script/generate_teacher_data.sh
```

Then, the data will be generated in `${YOUR_OUTPUT_DIR}/${dataset}/train_teacher_data` and `${YOUR_OUTPUT_DIR}/${dataset}/eval_teacher_data`, which are the parameters `train_data` and `valid_data` in `code/script/train.sh`.


2. Train Draft Model

Replace the parameters in `code/script/train.sh` with your own parameters, such as `LOG_DIR`, `OUTPUT_DIR`, `TARGET_MODEL`, `BASE_MODEL`, `MODEL_CLASS`, etc. And modify `accelerate.yaml` according to your needs if necessary.

```bash
LOG_DIR=YOUR_LOG_DIR
OUTPUT_DIR=YOUR_OUTPUT_DIR
TARGET_MODEL=YOUR_TARGET_MODEL_PATH
BASE_MODEL=YOUR_BASE_MODEL_PATH
MODEL_CLASS=AtSpeedRModel
```

Then, run the following command to train the target model.

```bash
cd code
bash script/train.sh
```


### Inference

First, replace the parameters in `code/script/inference.sh` with your own parameters, such as `LOG_DIR`, `OUTPUT_DIR`, `DRAFT_MODEL`, `DRAFT_MODEL_NAME`, etc.

```bash
LOG_DIR=YOUR_LOG_DIR
OUTPUT_DIR=YOUR_OUTPUT_DIR
DRAFT_MODEL=DRAFT_MODEL_PATH
DRATF_MODEL_NAME=DRAFT_MODEL_NAME
```

Then, run the following command to train the target model.

```bash
cd code
bash script/inference.sh
```


## Citation
If you find our work is useful for your research, please consider citing: 
```
@inproceedings{lin2024efficient,
  title={Efficient Inference for Large Language Model-based Generative Recommendation},
  author={Lin, Xinyu and Yang, Chaoqun and Wang, Wenjie and Li, Yongqi and Du, Cunxiao and Feng, Fuli and Ng, See-Kiong and Chua, Tat-Seng},
  booktitle={ICLR},
  year={2025}
}
```

## License

NUS © [NExT++](https://www.nextcenter.org/)