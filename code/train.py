import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
import argparse
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from utils import *
from model import *



def train(args, log=True):
    Model = eval(args.model_class)
    KD_args = {"KD_ratio": args.KD_ratio}
    if args.model_class == "AtSpeedSModel":
        KD_args.update({
            "topK": args.topK,
            "gamma": args.gamma,
            "r_original": args.r_original,
            "r_student": args.r_student,
        })
    elif args.model_class == "AtSpeedRModel":
        KD_args.update({
            "topK": args.topK,
            "gamma": args.gamma,
        })

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    gradient_accumulation_steps = args.train_batch_size // args.micro_batch_size if args.train_batch_size > args.micro_batch_size else 1
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if log and local_rank == 0:
        print('------------------ args:\n', vars(args))
        print('------------------', args.model_class, KD_args)

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size if gradient_accumulation_steps > world_size else 1

    # Check if parameter passed or if set within environ
    use_wandb = len(args.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if len(args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = args.wandb_watch

    # load tokenizer
    tokenizer_ckpt = args.target_model
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_ckpt, use_fast=True,legacy=True)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    if args.constrained_loss:
        train_data, valid_data = load_datasets(args)
        allowed_tokens_dict=valid_data.get_prefix_allowed_tokens_dict(tokenizer)
        KD_args.update({
            "allowed_tokens_dict": allowed_tokens_dict,
            "constrained_softmax": args.constrained_softmax,
            "temperature_softmax": args.temperature_softmax,
        })


    # load llama model
    model = Model.from_pretrained(
        args.base_model,
        # config=config,
        #load_in_8bit=True,
        # load_in_4bit=True,
        #torch_dtype=torch.float16,
        device_map=device_map,
        **KD_args,
    )
    model.resize_token_embeddings(len(tokenizer)) # , pad_to_multiple_of=1
    print('------------------ model loaded.')
    print('------------------------ device', device_map)

    if log and local_rank == 0:
        print(model)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    train_data, valid_data = load_dataset_with_teacher(args)

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.micro_batch_size,
            per_device_eval_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            fp16=args.fp16,
            # bf16=args.bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            # gradient_checkpointing=gradient_checkpointing,
            evaluation_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=2,
            load_best_model_at_end=True,
            # deepspeed=args.deepspeed,
            ddp_find_unused_parameters=False if ddp else None,
            eval_delay= 1 if args.save_and_eval_strategy=="epoch" else 2000,
            # report_to="wandb",
            # run_name=args.wandb_run_name,
            report_to=None,
            run_name=args.wandb_run_name,
            metric_for_best_model="eval_loss",  # metric for early stopping
            greater_is_better=False,  # whether greater value of metric is better
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
        tokenizer=tokenizer, # tiger
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8, 
            return_tensors="pt",
            padding="longest",
            max_length=512,
            # truncation=True,
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    # trainer.save_state()
    # trainer.save_model(output_dir=args.output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AtSpeed_train')
    parser = parse_global_args(parser)
    parser = parse_AtSpeed_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_llama_args(parser)
    args = parser.parse_args()

    train(args)
