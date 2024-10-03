from typing import Dict, List

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(gpu_name)

import os
import sys
import pandas as pd
from peft import PeftModel
from transformers import GenerationConfig,  LlamaTokenizer
from transformers import LlamaConfig
from transformers import LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from utils import *
from collator import TestCollator
from generation_trie import Trie

from torch.utils.data import DataLoader
import argparse

from tqdm import tqdm

import pickle

from beamSD.beamSD import *


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

def get_prefix_allowed_tokens_fn(indices, tokenizer):

    allowed_tokens = {}
    for index in indices.values():  # For each item
        for i, token in enumerate(index):  # For each code token at each position of the item
            token_id = tokenizer(token)["input_ids"][1]  # Get the token id of the code token, excluding the bos token
            if i not in allowed_tokens.keys():  # For each position i, create a dictionary if it doesn't exist
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

def parse_args(parser):
    parser.add_argument("--draft_base_model", type=str)
    parser.add_argument("--target_base_model", type=str)
    parser.add_argument("--target_ckpt_path", type=str)
    parser.add_argument("--draft_model_name", type=str)
    parser.add_argument("--target_model_name", type=str)
    parser.add_argument("--gamma", default=4, type=int)
    parser.add_argument("--run_beam_sizes", type=str)
    parser.add_argument("--draft_beam_size", type=int)
    parser.add_argument("--prefixAllowed", default=True, action="store_true")
    parser.add_argument("--do_sample", default=False, action="store_true")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--L", type=int, default=0)
    parser.add_argument("--R", type=int, default=None)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser = parse_llama_args(parser)
    parser = parse_args(parser)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))

    local_rank = 0
    device_map = {"": local_rank}
    device = torch.device("cuda",local_rank)

    load_8bit = True
    

    draft_base_model = args.draft_base_model
   
    tokenizer = LlamaTokenizer.from_pretrained(args.target_ckpt_path, legacy = True, use_fast=True) # load tokenizer
    tokenizer.padding_side = "left"

    draft_model = LlamaForCausalLM.from_pretrained(
        draft_base_model,
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
    prefixAllowed_suffix = "_greedy"  #"" if args.prefixAllowed else "noPrefixAllowed"
    do_sample_suffix = "do_sample" if args.do_sample else ""

    # greedy Trie
    # NOTE: test time costs uses this greedy Trie, for exp should use strict Trie
    # prefix_allowed_tokens = get_prefix_allowed_tokens_fn(test_data.indices, tokenizer)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=False, num_workers=0, pin_memory=True)

    def get_topk_results_by_score(predictions, scores, targets, k, all_items=None):
        results = []
        B = len(targets)
        predictions = [_.split("Response:")[-1] for _ in predictions]
        predictions = [_.strip().replace(" ","") for _ in predictions]
        if all_items is not None:
            for i, seq in enumerate(predictions):
                if seq not in all_items:
                    scores[i] = -1000
        batch_pred = []
        for b in range(B):
            batch_seqs = predictions[b * k: (b + 1) * k]
            batch_scores = scores[b * k: (b + 1) * k]
            pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
            results = sorted(pairs, key=lambda x: x[1], reverse=True)
            pred = [r[0] for r in results]
            batch_pred.append(pred)

        return batch_pred

    def get_topk_results(predictions, scores, targets, k, all_items=None):
        assert len(targets) == 1
        predictions = [_.split("Response:")[-1] for _ in predictions]
        predictions = [_.strip().replace(" ","") for _ in predictions]
        pred_allowed, pred_not_allowed = [], []
        if all_items is not None:
            for i, seq in enumerate(predictions):
                if seq not in all_items:
                    pred_not_allowed.append(seq)
                else:
                    pred_allowed.append(seq)
        pred = pred_allowed + pred_not_allowed
        batch_pred = [pred[:k]]
        return batch_pred

    get_topk_results = get_topk_results_by_score


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
                df_time_cost = pd.DataFrame(columns=['target_model', 'draft_model', 'beam_size', 'gamma', 'draft_time_cost', 'target_time_cost', 'loop process beam_scores time_cost', 'verify_time_cost', 'total_time_cost', 'generalBS_time_cost', 'TF_cache_time_cost', 'speedup', 'speedupTF', 
                                                 'ave_accept_rate', "total_accept_steps", "total_accept_tokens", "ave_accept_tokens", "ave_accept_tokens_rate", "overhead", "speedup'", "speedup'rate",
                                                'one_step_beam_search_tc', 
                                                'draft_forward_tc',
                                                'logits_processor_tc',
                                                'logits_processor_tc_per_beam',
                                                'target_forward_tc',
                                                'verify_logits_processor_tc',
                                                'verify_logits_processor_tc_per_step',
                                                'step1_draft_tc',
                                                'step1_target_tc',
                                                'step1_verify_tc',    
                                                 ])
                target_model.generation_config.num_beams = target_model.generation_config.num_return_sequences = beam_size

                for gamma in [args.gamma]:
                    print(f"draft: {draft_model_name} | target: {target_model_name} | target_beam_size: {beam_size} | draft_beam_size: {args.draft_beam_size} | gamma: {gamma} | prefixAllowed: {args.prefixAllowed} | do_sample: {args.do_sample} | sample_n: {stop_l}-{stop_r} | seed: {args.seed}")

                    all_pred_list, all_gold_list = [], []
                    all_pred_list_target, all_pred_list_TF_target = [], []
                    for step, batch in enumerate(tqdm(test_loader)):
                        if step < stop_l:
                            continue
                        elif step >= stop_r:
                            break

                        inputs = batch[0].to(device)
                        targets = batch[1]

                        if not flag:  # Warm-up for timing
                            target_model.generate(**inputs, use_cache=True, prefix_allowed_tokens_fn=prefix_allowed_tokens)
                            draft_model.generate(**inputs, use_cache=True, prefix_allowed_tokens_fn=prefix_allowed_tokens)
                            torch.cuda.synchronize(device)
                            flag = 1
                        outputs = BSSD(target_model, draft_model, inputs, gamma, max_new_tokens, prefix_allowed_tokens_fn=prefix_allowed_tokens)
                        target_outputs = target_generate(target_model, inputs, max_new_tokens, prefix_allowed_tokens_fn=prefix_allowed_tokens)
                        with Timer('TF_target') as t_TF:
                            TF_target_outputs = target_model.generate(**inputs, use_cache=True, prefix_allowed_tokens_fn=prefix_allowed_tokens, return_dict_in_generate=True, output_scores=True)
                        speedup = target_outputs['time_cost'] / outputs['time_cost']
                        speedupTF = t_TF.time_cost / outputs['time_cost']
                        overhead = (outputs['time_cost'] * max_new_tokens) / (target_outputs['time_cost'] * outputs['n_run'])
                        ave_accept_tokens_rate = outputs['ave_accept_rate'] * beam_size
                        speedup_ = outputs['ave_accept_tokens'] / overhead
                        speedup_rate = ave_accept_tokens_rate / overhead

                        df_time_cost.loc[len(df_time_cost)] = [
                            target_model_name, draft_model_name, beam_size, gamma, outputs['draft_time_cost'], outputs['target_time_cost'], outputs['loop process beam_scores time_cost'], outputs['verify_time_cost'], outputs['time_cost'], target_outputs['time_cost'], t_TF.time_cost, speedup, speedupTF,
                            outputs['ave_accept_rate'], outputs['total_accept_steps'], outputs['total_accept_tokens'], outputs['ave_accept_tokens'], ave_accept_tokens_rate, overhead, speedup_, speedup_rate,
                            outputs['one_step_beam_search_tc'],
                            outputs['draft_forward_tc'],
                            outputs['logits_processor_tc'],
                            outputs['logits_processor_tc_per_beam'],
                            outputs['target_forward_tc'],
                            outputs['verify_logits_processor_tc'],
                            outputs['verify_logits_processor_tc_per_step'],
                            outputs['step1_draft_tc'],
                            outputs['step1_target_tc'],
                            outputs['step1_verify_tc'],
                        ]
                        if beam_size == 20:
                            output = tokenizer.batch_decode(
                                outputs["beam_sequence"], skip_special_tokens=True
                            )
                            scores = outputs['beam_scores']
                            topk_res = get_topk_results(output, scores, targets, beam_size,
                                                        all_items=all_items if args.filter_items else None)
                            all_pred_list.extend(topk_res)
                            all_gold_list.extend(targets)

                            output_target = tokenizer.batch_decode(
                                target_outputs["beam_sequence"], skip_special_tokens=True
                            )
                            scores_target = target_outputs['beam_scores']
                            topk_res_target = get_topk_results(output_target, scores_target, targets, beam_size,
                                                        all_items=all_items if args.filter_items else None)
                            all_pred_list_target.extend(topk_res_target)
                            
                            output_TF_target = tokenizer.batch_decode(
                                TF_target_outputs["sequences"], skip_special_tokens=True
                            )
                            scores_TF_target = TF_target_outputs['sequences_scores']
                            topk_res_TF_target = get_topk_results(output_TF_target, scores_TF_target, targets, beam_size,
                                                        all_items=all_items if args.filter_items else None)
                            all_pred_list_TF_target.extend(topk_res_TF_target)

                    if beam_size == 20:
                        topN = [1, 3, 5, 10, 20]
                        test_results = computeTopNAccuracy(all_gold_list, all_pred_list, topN=topN, rank=local_rank)
                        print('--- BSSD results ---')
                        print_results(None, None, test_results)

                        test_results_target = computeTopNAccuracy(all_gold_list, all_pred_list_target, topN=topN, rank=local_rank)
                        print('--- my_target results ---')
                        print_results(None, None, test_results_target)

                        test_results_TF_target = computeTopNAccuracy(all_gold_list, all_pred_list_TF_target, topN=topN, rank=local_rank)
                        print('--- TF_target results ---')
                        print_results(None, None, test_results_TF_target)

                df_time_cost.groupby(['target_model', 'draft_model', 'beam_size', 'gamma']).mean().to_csv(f'AnaResult/{args.dataset}/timing_mean_{target_model_name}_{draft_model_name}_B{beam_size}-{args.draft_beam_size}_{stop_r}_{prefixAllowed_suffix}_{do_sample_suffix}_temp{args.temperature}_seed{args.seed}.csv')



