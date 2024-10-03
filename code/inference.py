from typing import Dict, List

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(gpu_name)

from torch.utils.data import DataLoader

from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, GenerationConfig
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from utils import *
from collator import TestCollator
from generation_trie import Trie
from beamSD import *



def get_prefix_allowed_tokens_fn(indices, tokenizer):

    allowed_tokens = {}
    for index in indices.values():  # For each item
        for i, token in enumerate(index):  # For each code token at each position of the item
            token_id = tokenizer(token)["input_ids"][1]  # Get the token id of the code token, excluding the bos token
            if i not in allowed_tokens.keys():  # For each position i, create a dictionary if it doesn"t exist
                allowed_tokens[i] = set()  # Create a dictionary for position i
            allowed_tokens[i].add(token_id)  # Add the generated token id of position i to the dictionary
    allowed_tokens[len(allowed_tokens.keys())] = set([tokenizer.eos_token_id])  # Add eos token at the end

    sep = tokenizer("Response:")["input_ids"][1:]

    def prefix_allowed_tokens_fn(batch_id, sentence):
        sentence = sentence.tolist()
        reversed_sent = sentence[::-1]
        for i in range(len(reversed_sent)):
            if reversed_sent[i:i + len(sep)] == sep[::-1]:
                return list(allowed_tokens[i])

    return prefix_allowed_tokens_fn



def inference(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))

    local_rank = 0
    device_map = {"": local_rank}
    device = torch.device("cuda",local_rank)

    load_8bit = True
    
    draft_model = args.draft_model
   
    tokenizer = LlamaTokenizer.from_pretrained(args.target_ckpt_path, legacy = True, use_fast=True) # load tokenizer
    tokenizer.padding_side = "left"

    draft_model = LlamaForCausalLM.from_pretrained(
        draft_model,
        #load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    set_seed(42)  # A mysterious bug in transformers, must set_seed(42) here, otherwise model output will change even if do_sample=False
    draft_model.resize_token_embeddings(len(tokenizer))
    target_base_model = args.target_base_model
    target_ckpt_path = args.target_ckpt_path
    target_lora_weights = target_ckpt_path
    target_model = LlamaForCausalLM.from_pretrained(
        target_base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    set_seed(42)  # A mysterious bug in transformers, must set_seed(42) here, otherwise model output will change even if do_sample=False
    target_model.resize_token_embeddings(len(tokenizer))
    set_seed(args.seed)
    target_model = PeftModel.from_pretrained(
        target_model,
        target_lora_weights,
        torch_dtype=torch.float16,
        device_map=device_map
    )
    print("** model loaded")

    # unwind broken decapoda-research config
    for model in [draft_model, target_model]:
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.config.return_dict = True
        model.generation_config.return_dict_in_generate = True
        model.config.use_cache = True

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.
        
        model.config.decoder_start_token_id = 1

        model.eval()
    
    prompt_ids = [0]

    test_data = load_test_dataset(args)
    n = len(test_data)
    stop_l = args.L
    stop_r = min(n, args.R) if args.R != None else n

    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    # strict Trie
    candidate_trie = Trie([tokenizer.encode("{}".format(e))[:]+[tokenizer.eos_token_id] for e in all_items])
    prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)
    do_sample_suffix = "do_sample" if args.do_sample else ""

    # greedy Trie
    # NOTE: test time costs uses this greedy Trie, for exp should use strict Trie
    # prefix_allowed_tokens = get_prefix_allowed_tokens_fn(test_data.indices, tokenizer)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=False, num_workers=0, pin_memory=True)

    with torch.no_grad():
        for prompt_id in prompt_ids:
            test_loader.dataset.set_prompt(prompt_id)

            flag = 0  # Warm-up for timing
            draft_model_name, target_model_name = args.draft_model_name, args.target_model_name
            max_new_tokens = target_model.generation_config.max_new_tokens = draft_model.generation_config.max_new_tokens = 4
            draft_model.generation_config.num_beams = draft_model.generation_config.num_return_sequences = args.draft_beam_size
            target_model.generation_config.do_sample = draft_model.generation_config.do_sample = args.do_sample
            target_model.generation_config.temperature = draft_model.generation_config.temperature = args.temperature
            for beam_size in eval(args.run_beam_sizes):  # [1, 3, 5, 10, 20]:
                df_time_cost = pd.DataFrame(columns=[
                    "target_model", "draft_model", "beam_size", "gamma", 
                    "draft_time_cost", "target_time_cost", "verify_time_cost", "total_time_cost", "generalBS_time_cost", "TF_cache_time_cost", 
                    "speedup", "speedupTF", "total_accept_steps", "total_accept_tokens", "ave_accept_tokens", "overhead",
                ])
                target_model.generation_config.num_beams = target_model.generation_config.num_return_sequences = beam_size

                for gamma in [args.gamma]:
                    print(f"target: {target_model_name} | draft: {draft_model_name} | target_beam_size: {beam_size} | draft_beam_size: {args.draft_beam_size} | gamma: {gamma} | do_sample: {args.do_sample} | sample_n: {stop_l}-{stop_r} | seed: {args.seed}")

                    for step, batch in enumerate(tqdm(test_loader)):
                        if step < stop_l:
                            continue
                        elif step >= stop_r:
                            break

                        inputs = batch[0].to(device)

                        if not flag:  # Warm-up for timing
                            target_model.generate(**inputs, use_cache=True, prefix_allowed_tokens_fn=prefix_allowed_tokens)
                            draft_model.generate(**inputs, use_cache=True, prefix_allowed_tokens_fn=prefix_allowed_tokens)
                            torch.cuda.synchronize(device)
                            flag = 1
                        outputs = BSSD(target_model, draft_model, inputs, gamma, max_new_tokens, prefix_allowed_tokens_fn=prefix_allowed_tokens)
                        target_outputs = target_generate(target_model, inputs, max_new_tokens, prefix_allowed_tokens_fn=prefix_allowed_tokens)
                        with Timer("TF_target") as t_TF:
                            TF_target_outputs = target_model.generate(**inputs, use_cache=True, prefix_allowed_tokens_fn=prefix_allowed_tokens, return_dict_in_generate=True, output_scores=True)
                        speedup = target_outputs["time_cost"] / outputs["time_cost"]
                        speedupTF = t_TF.time_cost / outputs["time_cost"]
                        overhead = (outputs["time_cost"] * max_new_tokens) / (target_outputs["time_cost"] * outputs["n_run"])

                        df_time_cost.loc[len(df_time_cost)] = [
                            target_model_name, draft_model_name, beam_size, gamma, 
                            outputs["draft_time_cost"], outputs["target_time_cost"], outputs["verify_time_cost"], outputs["time_cost"], target_outputs["time_cost"], t_TF.time_cost, 
                            speedup, speedupTF, outputs["total_accept_steps"], outputs["total_accept_tokens"], outputs["ave_accept_tokens"], overhead,
                        ]
                        
                df_time_cost.groupby(["target_model", "draft_model", "beam_size", "gamma"]).mean().to_csv(f"AnaResult/{args.dataset}/timing_mean_{target_model_name}_{draft_model_name}_B{beam_size}-{args.draft_beam_size}_{stop_l}-{stop_r}_{do_sample_suffix}_temp{args.temperature}_seed{args.seed}.csv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AtSpeed_inference")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser = parse_llama_args(parser)
    parser = parse_inference_args(parser)
    args = parser.parse_args()
    inference(args)