import os
import sys
import numpy as np
import torch
import argparse
import copy
import pickle
import psutil

from typing import List, Dict
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import GenerationConfig
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

from utils import *


def bytes_to_gb(bytes, round_digits=2):
    # Convert bytes to GB and round to the specified number of decimal places
    return round(bytes / (1024 ** 3), round_digits)

class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
            
    def __call__(self, batch):
        input_texts = [d["input_ids"] + d["labels"]+ self.tokenizer.eos_token for d in batch]
        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=512,
                                truncation=True,
                                return_attention_mask=True)

        inputs['labels'] = copy.deepcopy(inputs["input_ids"])
        
        return inputs



def main(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    beam_size = args.beam_size

    device = torch.device("cuda", 0)
    device_map = "auto"
    base_model = args.base_model
    lora_weights = args.target_model

    tokenizer = LlamaTokenizer.from_pretrained(lora_weights,use_fast=True,legacy=True)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16, # torch_dtype=torch.bfloat16
        device_map=device_map,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16, # torch_dtype=torch.bfloat16
        device_map=device_map
    )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.config.decoder_start_token_id = 1
    model.config.use_cache = False

    collator = Collator(args, tokenizer)

    def prefix_allowed_tokens_fn(candidate_trie):

        sep = tokenizer("Response:")["input_ids"][1:]
        bos = [1]
        def prefix_allowed_tokens(batch_id, sentence):
            for i in range(len(sentence),-1,-1):
                if sentence[i-len(sep):i].tolist() == sep:
                    if i == len(sentence):
                        sentence_ = bos
                    else:
                        sentence_ = [1] + sentence[i:].tolist()
            trie_out = candidate_trie.get(sentence_)
            return trie_out

        return prefix_allowed_tokens

    def get_logits(dataset, mode="train"):

        data_loader = DataLoader(dataset, batch_size=args.micro_batch_size, 
                              collate_fn=collator, num_workers=2, pin_memory=True)

        all_items = load_test_dataset(args).get_all_items()

        # strict Trie
        candidate_trie = Trie([tokenizer.encode("{}".format(e))[:]+[tokenizer.eos_token_id] for e in all_items])
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

        generation_config = GenerationConfig(
            temperature=1,
            do_sample=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            output_hidden_states=True,
            output_attentions=True
        )

        def get_output(beam_size,**inputs):
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_token,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
                generation_config=generation_config
            )
            output_ids = output["sequences"]
            return output_ids[...,-5:] # TIGER/LC-Rec
        
        flag = 1
        split_idx = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(data_loader)):
                inputs = {k: v.to(device) for k, v in batch.items()}

                # 0. original output teacher logits (every position)
                outputs = model(**inputs)
                batch['teacher_logits'] = outputs.logits[:, -6:-1].cpu()

                # 1. teacher output beam sequence 
                generate_input = {k:v[...,:-5] for k,v in inputs.items()}
                batch['teacher_output'] = get_output(beam_size, **generate_input) # get the beam output generated by the teacher model

                # 2. teacher output beam sequence logits (every position) - ok
                model_output_input = {k:torch.cat([v.repeat(20,1),batch['teacher_output']],dim=1) for k,v in generate_input.items()}
                outputs = model(**model_output_input)
                batch['teacher_output_logits'] = outputs.logits[:, -6:-1, 32000:].cpu()
                
                # Build a new first dimension to represent a sample for later unbind(v) to separate each sample
                batch['teacher_output'] = batch['teacher_output'].unsqueeze(0)
                batch['teacher_output_logits'] = batch['teacher_output_logits'].unsqueeze(0) 
                
                if flag:
                    result = {k:tuple() for k in batch.keys()}
                    flag = 0
                    
                for k, v in batch.items():
                    result[k] += torch.unbind(v)

                if (step+1) % 2000 == 0:
                    # access memeory utility
                    memory = psutil.virtual_memory()
                    if memory.percent > 80:
                        print(f"Total memory: {bytes_to_gb(memory.total)} GB")
                        print(f"Used memory: {bytes_to_gb(memory.used)} GB")
                        print(f"Memory usage percentage: {memory.percent}%")
                        torch.save(result, f"{args.output_dir}/teacher_output_info_split/output_{split_idx}.pt")
                        flag = 1
                        split_idx += 1

                if mode == "valid":
                    if (step+1) == 2000:
                        break
        return result, split_idx
    
    train_data, valid_data = load_datasets(args)
    results_train, split_idx = get_logits(train_data, "train")
    torch.save(results_train, f"{args.output_dir}/train_teacher_info.pt")

    results_valid, split_idx = get_logits(valid_data, "valid")
    torch.save(results_valid, f"{args.output_dir}/valid_teacher_info.pt")

    
    train_save_dir = f"{args.output_dir}/train_teacher_data"
    valid_save_dir = f"{args.output_dir}/valid_teacher_data"
    valid_data = Dataset.from_dict(results_valid)
    valid_data.save_to_disk(valid_save_dir)

    split_len = 5000

    def batch(data, batch_size=split_len):
        data_len = len(data['input_ids'])
        chunk_size = (data_len - 1) // batch_size + 1
        for i in range(chunk_size):
            yield {k:v[batch_size * i: batch_size * (i + 1)] for k,v in data.items()}
            
    for idx, split in enumerate(batch(results_train)):
        print(split.keys())
        print(len(split['input_ids']))
        split = Dataset.from_dict(split)
        split.save_to_disk(f"{train_save_dir}/train_teacher_data_{idx}")
    print(f"--- train_teacher_data is saved in {train_save_dir}")
    print(f"--- eval_teacher_data is saved in {valid_save_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_test_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_llama_args(parser)
    parser.add_argument("--beam_size", type=int, default=20, help="The beam size of target model to generate")
    args = parser.parse_args()

    main(args)
