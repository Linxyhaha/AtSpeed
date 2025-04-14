# AtSpeed
This is the pytorch implementation of our paper
> Efficient Inference for Large Language Model-based Generative Recommendation

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

Then data will be generated in `${YOUR_OUTPUT_DIR}/${dataset}/train_teacher_data` and `${YOUR_OUTPUT_DIR}/${dataset}/eval_teacher_data`, which is the `train_data` and `valid_data` parameter in `code/script/train.sh`.


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