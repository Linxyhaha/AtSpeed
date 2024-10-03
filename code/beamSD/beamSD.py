import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import time
import torch.nn.functional as F
from typing import Dict, Optional, List, Callable
from transformers.generation.logits_process import LogitsProcessorList, LogitsProcessor
from transformers import set_seed



class Timer:  # Timer for code block or function
    def __init__(self, func="", sync_cuda=True, syn_device=0):
        self.func = func
        self.sync_cuda = sync_cuda
        self.syn_device = syn_device
    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sync_cuda:
            torch.cuda.synchronize(self.syn_device)  # For precise timing
        self.time_cost = time.time() - self.start

    def __call__(self, func):
        def wrapper(*args, syn_device=self.syn_device, **kwargs):
            self.syn_device = syn_device
            self.start = time.time()
            result = func(*args, **kwargs)
            if self.sync_cuda:
                torch.cuda.synchronize(self.syn_device)
            self.time_cost = time.time() - self.start
            result["time_cost"] = self.time_cost
            return result
        return wrapper


@Timer()
@torch.no_grad()
def one_step_beam_search(
    model, 
    inputs: Dict, 
    beam_size: int,
    beam_scores: torch.FloatTensor, 
    beam_sequence: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
) -> Dict:
    with Timer('draft_model.forward') as timer0:
        outputs = model(**inputs)
    do_sample = model.generation_config.do_sample
    n_last_beam = len(beam_scores)
    if len(logits_processor) != 0:
        logits_processor[0]._num_beams = n_last_beam
    next_token_logits = outputs.logits[0, -n_last_beam:]  # [batch_size * beam_size, vocab_size]
    next_token_scores = F.log_softmax(next_token_logits, dim=-1)
    # Enter logits_processor and pass it to prefix_allowed_tokens_fn. It requires the entire sequence's input_ids and should be converted to [beam_size, len_seq, vocab_size] format.
    with Timer('logits_processor') as timer1:
        if n_last_beam == 1 and beam_size != 1:  # For fist step
            next_token_scores = logits_processor(beam_sequence, next_token_scores.repeat(beam_size, 1))[:1]
        else:
            next_token_scores = logits_processor(beam_sequence, next_token_scores)
    if do_sample:
        next_token_scores = logits_warper(beam_sequence, next_token_scores)

    vocab_size = next_token_scores.shape[-1]
    beam_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
    beam_scores = beam_scores.view(-1)
    if do_sample:
        probs = F.softmax(beam_scores, dim=-1)  # Probabilities of all sequences
        next_tokens = torch.multinomial(probs, num_samples=beam_size)
        beam_scores = torch.gather(beam_scores, -1, next_tokens)
    else:
        beam_scores, next_tokens = beam_scores.topk(beam_size)
    beam_indices = next_tokens // vocab_size
    beam_tokens = next_tokens % vocab_size
    # process not allowed tokens (filter tokens generated outside the range based on topk + topk)
    if len(logits_processor) != 0:
        beam_tokens_allowed = (beam_tokens >= 32000) | (beam_tokens == 2)
        next_tokens = next_tokens[beam_tokens_allowed]
        beam_scores = beam_scores[beam_tokens_allowed]
        beam_indices = beam_indices[beam_tokens_allowed]
        beam_tokens = beam_tokens[beam_tokens_allowed]
        beam_size = beam_tokens.shape[0]
    beam_sequence = torch.cat((beam_sequence[beam_indices], beam_tokens[:, None]), dim=-1)
    min_dtype = torch.finfo(model.dtype).min
    causal_mask = torch.cat((inputs["attention_mask"][0, 0, -n_last_beam:][beam_indices], (torch.eye(beam_size, device=model.device) == 0) * min_dtype), dim=-1)
    causal_mask = causal_mask[None, None, :, :]
    position_ids = (inputs["position_ids"][:, -1:] + 1).repeat(1, beam_size)

    return {
        "probs": probs if do_sample else None,
        "seq_tokens": next_tokens,
        "beam_sequence": beam_sequence,  # [beam_size, len_sequence_prefix + 1]
        "beam_scores": beam_scores,  # [beam_size,]
        "beam_indices": beam_indices,  # [beam_size,]
        "beam_tokens": beam_tokens,  # [beam_size,]
        "attention_mask": causal_mask,  # [1, 1, beam_size, len_prefix + beam_size]
        "position_ids": position_ids,  # [1, beam_size]
        "past_key_values": outputs.past_key_values,  # tuple(n_layers, 2, tensor[1, n_heads, len_prefix, head_dim])
        "draft_model.forward time_cost": timer0.time_cost,
        "logits_processor time_cost": timer1.time_cost,
        "logits_processor time_cost (per beam)": timer1.time_cost / n_last_beam,
    }

@torch.no_grad()
def _draft_beam_search(
    model, 
    inputs: Dict, 
    draft_len: int, 
    beam_size: int,
    beam_scores: torch.FloatTensor, 
    beam_sequence: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
) -> Dict:
    # prepare outputs for verify
    step_beam_sequence = (beam_sequence,)
    step_probs = tuple()
    step_seq_tokens = tuple()
    step_beam_indices= tuple()
    step_beam_tokens = tuple()
    step_attention_mask = tuple()
    step_position_ids = tuple()
    step_len = [len(beam_scores)]
    min_dtype = torch.finfo(model.dtype).min
    len_prefix, len_seq_prefix = inputs["attention_mask"].shape[2:]

    # forward
    one_step_beam_search_tc = []
    draft_forward_tc = []
    logits_processor_tc = []
    logits_processor_tc_per_beam = []
    for i in range(draft_len):
        outputs = one_step_beam_search(model, inputs, beam_size, beam_scores, beam_sequence, logits_processor, logits_warper)
        inputs = {
            "input_ids": outputs["beam_tokens"][None, :],
            "attention_mask": outputs["attention_mask"],
            "position_ids": outputs["position_ids"],
            "past_key_values": outputs["past_key_values"]
        }
        beam_scores = outputs["beam_scores"]
        beam_sequence = outputs["beam_sequence"]
        step_len.append(len(beam_scores))
        step_beam_sequence += (beam_sequence,)
        step_probs += (outputs["probs"],)
        step_seq_tokens += (outputs["seq_tokens"],)
        step_beam_indices += (outputs["beam_indices"],)
        step_beam_tokens += (outputs["beam_tokens"],)
        step_attention_mask += (outputs["attention_mask"],)
        step_position_ids += (outputs["position_ids"],)
        one_step_beam_search_tc.append(outputs["time_cost"])
        draft_forward_tc.append(outputs["draft_model.forward time_cost"])
        logits_processor_tc.append(outputs["logits_processor time_cost"])
        logits_processor_tc_per_beam.append(outputs["logits_processor time_cost (per beam)"])
    draft_past_key_values = outputs["past_key_values"]
    one_step_beam_search_tc = sum(one_step_beam_search_tc) / draft_len if draft_len != 0 else 0
    draft_forward_tc = sum(draft_forward_tc) / draft_len if draft_len != 0 else 0
    logits_processor_tc = sum(logits_processor_tc) / draft_len if draft_len != 0 else 0
    logits_processor_tc_per_beam = sum(logits_processor_tc_per_beam) / draft_len if draft_len != 0 else 0

    return {
        "step_len": step_len,  # [1 + draft_len,]
        "step_probs": step_probs,  # [draft_len, beam_size * vocab_size]
        "step_seq_tokens": step_seq_tokens,  # [draft_len, beam_size]
        "step_beam_sequence": step_beam_sequence,  # (draft_len, [beam_size, len_sequence_prefix + i])
        "beam_scores": beam_scores,  # [beam_size,]
        "step_beam_indices": step_beam_indices,  # [draft_len, beam_size]
        "step_beam_tokens": step_beam_tokens,  # [draft_len, beam_size]
        "step_attention_mask": step_attention_mask,  # (draft_len, [1, 1, beam_size, len_sequence_prefix + i])
        "step_position_ids": step_position_ids,  # (draft_len, [1, beam_size])
        "past_key_values": draft_past_key_values,  # (n_layers, 2, [1, n_heads, len_sequence_prefix + beam_size * (draft_len - 1), head_dim])
        "one_step_beam_search time_cost": one_step_beam_search_tc,
        "draft_forward time_cost": draft_forward_tc,
        "logits_processor time_cost": logits_processor_tc,
        "logits_processor time_cost (per beam)": logits_processor_tc_per_beam,
    }

@Timer()
def draft_beam_search(*args, **kwargs):
    return _draft_beam_search(*args, **kwargs)

@Timer(sync_cuda=False)
def no_delay_draft_beam_search(*args, **kwargs):
    return _draft_beam_search(*args, **kwargs)


@torch.no_grad()
def _target_beam_search(
    model,  
    inputs: Dict, 
    draft_outputs: Dict, 
    beam_size: int,
    beam_scores: torch.FloatTensor, 
    beam_sequence: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> Dict:
    min_dtype = torch.finfo(model.dtype).min
    draft_len = len(draft_outputs["step_beam_indices"])
    # prepare inputs
    input_ids = torch.cat((inputs["input_ids"][0],) + draft_outputs["step_beam_tokens"], dim=-1)[None, :]
    step_attention_mask = (inputs["attention_mask"],) + draft_outputs["step_attention_mask"]
    n = max(mask.shape[3] for mask in step_attention_mask)
    attention_mask = torch.cat(
    [torch.cat((mask, torch.full((1, 1, mask.shape[2], n - mask.shape[3]), min_dtype, device=mask.device)), dim=3) 
    for mask in step_attention_mask], 
    dim=2)
    step_position_ids = (inputs["position_ids"],) + draft_outputs["step_position_ids"]
    position_ids = torch.cat(step_position_ids, dim=1)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": inputs["past_key_values"]
    }
    # forward
    with Timer('target_model.forward') as timer:
        outputs = model(**inputs)

    n_last_beam = len(beam_scores)
    next_token_scores = outputs.logits[0, -sum(draft_outputs["step_len"]):]

    return {
        "next_token_scores": next_token_scores,  # [target_len * beam_size, vocab_size]
        "attention_mask": attention_mask,
        "position_ids": position_ids,  # [1, len_sequence_prefix + beam_size * target_len]
        "past_key_values": outputs.past_key_values,  # (n_layers, 2, [1, n_heads, len_sequence_prefix + beam_size * (target_len - 1), head_dim])
        "target_forward time_cost": timer.time_cost
    }

@Timer()
def target_beam_search(*args, **kwargs):
    return _target_beam_search(*args, **kwargs)

@Timer(sync_cuda=False)
def no_delay_target_beam_search(*args, **kwargs):
    return _target_beam_search(*args, **kwargs)

@Timer()
def verify(
    target_model,
    target_model_inputs: Dict, 
    draft_outputs: Dict, 
    target_outputs: Dict,
    draft_beam_size: int,
    beam_size: int,
    beam_scores: torch.FloatTensor, 
    beam_sequence: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
) -> Dict:
    do_sample = target_model.generation_config.do_sample
    first = target_model_inputs['past_key_values'] is None  # Whether step1 or not
    draft_len = len(draft_outputs["step_beam_indices"])
    target_len = draft_len + 1  # gamma + 1
    len_sequence_prefix = target_model_inputs["attention_mask"].shape[-1]
    len_prefix = target_model_inputs["input_ids"].shape[-1]
    device = target_model_inputs["attention_mask"].device
    min_dtype = torch.finfo(target_model_inputs["attention_mask"].dtype).min

    draft_step_beam_sequence = draft_outputs["step_beam_sequence"]

    step_len = draft_outputs["step_len"]
    next_token_scores = target_outputs["next_token_scores"]
    vocab_size = next_token_scores.shape[-1]
    causal_mask = target_outputs["attention_mask"]
    input_mask = causal_mask[0, 0]
    input_position_ids = target_outputs["position_ids"]
    input_position_ids = torch.cat((input_position_ids, (input_position_ids[:, -1:] + 1).repeat(1, step_len[-1])), dim=-1)

    n_matches = 0
    logits_processor_tc = []
    with Timer('loop process beam_scores') as timer_loop:
        i_l, i_r = 0, step_len[0]
        for i in range(draft_len+1):
            scores = next_token_scores[i_l: i_r]
            if len(logits_processor) != 0:
                logits_processor[0]._num_beams = beam_size
            if n_matches != draft_len:
                i_l, i_r = i_r, i_r + step_len[i+1]
            scores = scores[hit_indices] if i > 0 else scores
            scores = F.log_softmax(scores, dim=-1)
            with Timer('logits_processor') as timer:
                if first and i == 0 and beam_size != 1:  # When step1
                    scores = logits_processor(beam_sequence, scores.repeat(beam_size, 1))[:1]
                else:
                    draft_beam_sequence = draft_step_beam_sequence[i][hit_indices] if i > 0 else draft_step_beam_sequence[i]
                    scores = logits_processor(draft_beam_sequence, scores)
            logits_processor_tc.append(timer.time_cost)
            if do_sample:
                scores = logits_warper(None, scores)
            if i > 0 and not do_sample:
                beam_scores = beam_scores[hit_indices4beam_scores]
            beam_scores = scores + beam_scores[:, None].expand_as(scores)
            beam_scores = beam_scores.view(-1)
            if do_sample:
                probs = F.softmax(beam_scores, dim=-1)
                if n_matches == draft_len:
                    next_tokens = torch.multinomial(probs, num_samples=beam_size)
                    beam_scores = torch.gather(beam_scores, -1, next_tokens)
                    beam_indices = next_tokens // vocab_size
                    beam_tokens = next_tokens % vocab_size
                    beam_indices = hit_indices[beam_indices]
                    break
                draft_probs = draft_outputs["step_probs"][i]
                if i > 0:  # Map probs to the draft's probability space for comparison
                    draft_probs = draft_probs.view(-1, vocab_size)
                    beam_scores = beam_scores.view(-1, vocab_size)
                    target_beam_scores = torch.full_like(draft_probs, -float('inf'))
                    target_beam_scores[hit_indices] = beam_scores
                    target_beam_scores = target_beam_scores.view(-1)
                    target_probs = F.softmax(target_beam_scores, dim=-1)
                    probs = target_probs
                    draft_probs = draft_probs.view(-1)
                    beam_scores = target_beam_scores.view(-1)
                    # Handle nan caused by -inf
                    probs[torch.isnan(probs)] = 0
                    draft_probs[torch.isnan(draft_probs)] = 0
            else:
                beam_scores, next_tokens = beam_scores.topk(beam_size)
                beam_indices = next_tokens // vocab_size
                beam_tokens = next_tokens % vocab_size
                if i > 0:  # Map probs to the draft's probability space for comparison
                    beam_indices = hit_indices[beam_indices]
                    next_tokens = beam_indices * vocab_size + beam_tokens
                if n_matches == draft_len:
                    break
            # verify
            if do_sample:
                q = draft_probs
                p = probs
                q_i = q[draft_outputs["step_seq_tokens"][i]]
                p_i = p[draft_outputs["step_seq_tokens"][i]]
                probability_ratio = p_i / q_i
                r_i = torch.rand_like(probability_ratio)
                is_accepted = r_i <= probability_ratio
                accepted_seq_tokens = draft_outputs["step_seq_tokens"][i][is_accepted]
                if is_accepted.sum() >= beam_size:  # Accept. Continue.
                    n_matches += 1
                    accepted_indices = torch.randperm(len(accepted_seq_tokens))[:beam_size]
                    seq_tokens = accepted_seq_tokens[accepted_indices]
                    seq_tokens = seq_tokens.sort()[0]  # 保持 draft 的顺序
                    hit_indices = torch.cat([torch.where(draft_outputs["step_seq_tokens"][i]==y)[0] for y in seq_tokens])
                    beam_scores = torch.gather(beam_scores, -1, seq_tokens)
                    beam_indices = seq_tokens // vocab_size
                    beam_tokens = seq_tokens % vocab_size
                else:  # Reject. Resample.
                    new_probs = torch.clamp((p - q), min=0)
                    new_probs[accepted_seq_tokens] = 0
                    if new_probs.sum() == 0:
                        if i > 0:
                            new_probs = new_probs.view(-1, vocab_size)
                            target_probs = new_probs[hit_indices]
                            new_probs[hit_indices][target_probs == 0] = torch.finfo(new_probs.dtype).tiny
                            new_probs = new_probs.view(-1)
                        else:
                            new_probs[...] = torch.finfo(new_probs.dtype).tiny
                    else:
                        new_probs = new_probs / new_probs.sum()
                    next_tokens = torch.multinomial(new_probs, num_samples=beam_size - is_accepted.sum())
                    seq_tokens = torch.cat((accepted_seq_tokens, next_tokens))
                    seq_tokens = seq_tokens.sort()[0]  # Maintain the order of the draft
                    beam_scores = torch.gather(beam_scores, -1, seq_tokens)
                    beam_indices = seq_tokens // vocab_size
                    beam_tokens = seq_tokens % vocab_size
                    break
            else:
                draft_seq_tokens = draft_outputs["step_seq_tokens"][i].tolist()
                target_seq_tokens = next_tokens.tolist()
                hit_indices = [k for k, token in enumerate(draft_seq_tokens) if token in target_seq_tokens]
                hit_indices = torch.tensor(hit_indices, device=device)
                hit_indices4beam_scores = torch.cat([torch.where(draft_outputs["step_seq_tokens"][i]==y)[0] for y in target_seq_tokens])
                hit_indices4beam_scores = hit_indices4beam_scores.sort()[1]
                if len(hit_indices) == beam_size:
                    n_matches += 1
                else:
                    break
    logits_processor_tc = sum(logits_processor_tc)
    logits_processor_tc_per_step = logits_processor_tc / (i + 1)
    beam_sequence = torch.cat((draft_step_beam_sequence[n_matches][beam_indices, :], beam_tokens[:, None]), dim=-1)
    if first:
        target_step_len = [len_sequence_prefix - 1] + draft_outputs["step_len"] + [beam_size]
    else:
        target_step_len = [0] + draft_outputs["step_len"] + [beam_size]
    step_len_seq = torch.tensor(target_step_len).cumsum(dim=0)
    if first:
        n_seq_tokens = step_len_seq[n_matches+1]
    else:
        n_seq_tokens = len_sequence_prefix + step_len_seq[n_matches+1]
    # prepare inputs
    input_ids = beam_tokens[None, :]
    if first:
        mask = input_mask[step_len_seq[n_matches]: step_len_seq[n_matches+1], :step_len_seq[n_matches+1]][beam_indices]
    else:
        mask = input_mask[step_len_seq[n_matches]: step_len_seq[n_matches+1], :len_sequence_prefix + step_len_seq[n_matches+1]][beam_indices]
    causal_mask = torch.cat((mask, torch.full((beam_size, beam_size), min_dtype, device=device).fill_diagonal_(0)), dim=-1)
    attention_mask = causal_mask[None, None, :, :]
    position_ids = input_position_ids[:, step_len_seq[n_matches+1]: step_len_seq[n_matches+2]][:, :beam_size]
    if n_matches == draft_len:
        last_n_matches = n_matches - 1
        draft_input_ids = torch.cat((draft_outputs["step_beam_tokens"][last_n_matches], beam_tokens), dim=-1)[None, :]
        if first:
            last_mask = input_mask[step_len_seq[n_matches]: step_len_seq[n_matches+1], :step_len_seq[n_matches+1]]
        else:
            last_mask = input_mask[step_len_seq[n_matches]: step_len_seq[n_matches+1], :len_sequence_prefix + step_len_seq[n_matches+1]]
        draft_causal_mask = torch.cat(
            (torch.cat((last_mask, mask), dim=0),
             torch.full((draft_input_ids.shape[1], beam_size), min_dtype, device=device))
            , dim=-1)
        for i in range(-beam_size, 0):
            draft_causal_mask[i, i] = 0
        draft_attention_mask = draft_causal_mask[None, None, :, :]
        draft_position_ids = input_position_ids[:, -draft_input_ids.shape[1]:]

    target_outputs["past_key_values"] = list(target_outputs["past_key_values"])
    for i in range(len(target_outputs["past_key_values"])):
        target_outputs["past_key_values"][i] = (
            target_outputs["past_key_values"][i][0][:, :, :n_seq_tokens, :],
            target_outputs["past_key_values"][i][1][:, :, :n_seq_tokens, :],
        )
    draft_outputs["past_key_values"] = list(draft_outputs["past_key_values"])
    for i in range(len(draft_outputs["past_key_values"])):
        draft_outputs["past_key_values"][i] = (
            draft_outputs["past_key_values"][i][0][:, :, :n_seq_tokens, :],
            draft_outputs["past_key_values"][i][1][:, :, :n_seq_tokens, :],
        )
    target_model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": target_outputs["past_key_values"]
    }
    draft_model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": draft_outputs["past_key_values"]
    }
    if n_matches == draft_len:
        draft_model_inputs["input_ids"] = draft_input_ids
        draft_model_inputs["attention_mask"] = draft_attention_mask
        draft_model_inputs["position_ids"] = draft_position_ids

    return {
        "n_matches": n_matches,
        "beam_sequence": beam_sequence,  # [beam_size, cur_len]
        "beam_scores": beam_scores,
        "target_model_inputs": target_model_inputs,
        "draft_model_inputs": draft_model_inputs,
        "loop process beam_scores time_cost": timer_loop.time_cost,
        "logits_processor time_cost": logits_processor_tc,
        "logits_processor time_cost (per step)": logits_processor_tc_per_step,
    }

@Timer()
@torch.no_grad()
def BSSD(
    target_model, 
    draft_model, 
    inputs: Dict,
    gamma: int, 
    max_new_tokens: int, 
    logits_processor: Optional[LogitsProcessorList] = None,
    prefix_allowed_tokens_fn = None,
) -> Dict:
    # initialize
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    logits_processor = target_model._get_logits_processor(
        generation_config=target_model.generation_config,
        input_ids_seq_length=inputs["input_ids"].shape[-1],
        encoder_input_ids=inputs["input_ids"],
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs["input_ids"].device,
    )
    logits_warper = (
        target_model._get_logits_warper(target_model.generation_config) if target_model.generation_config.do_sample else None
    )
    beam_size = target_model.generation_config.num_beams
    draft_beam_size = draft_model.generation_config.num_beams
    cur_len = inputs["input_ids"].shape[-1]
    max_len = cur_len + max_new_tokens
    beam_sequence = inputs["input_ids"]
    n = inputs["input_ids"].shape[-1]
    min_dtype = torch.finfo(draft_model.dtype).min
    causal_mask = (torch.tril(torch.ones((n, n), device=draft_model.device)) == 0) * min_dtype
    inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": causal_mask[None, None, :, :],
        "position_ids": torch.arange(n, device=draft_model.device).unsqueeze(0),
        "past_key_values": None
    }
    draft_model_inputs = target_model_inputs = inputs
    # Since logits_processor internally keeps track of beam_size, the input beam_scores and beam_sequence must be in batch form
    beam_scores = torch.zeros(inputs["input_ids"].shape[0], dtype=draft_model.dtype, device=draft_model.device)
    beam_sequence = inputs["input_ids"].repeat(beam_size, 1)

    accept_rates, draft_time_cost, target_time_cost, verify_time_cost = [], 0, 0, 0
    lp_time_cost = 0
    accept_steps = []
    # BSSD
    while cur_len < max_len:
        draft_len = min(gamma, max_len - cur_len - 1)
        if draft_len == 0:
            verify_outputs = one_step_beam_search(target_model, target_model_inputs, beam_size, beam_scores, beam_sequence, logits_processor, logits_warper)
            beam_sequence = verify_outputs["beam_sequence"]
            beam_scores = verify_outputs["beam_scores"]
            break
        # 1. draft
        draft_outputs = draft_beam_search(draft_model, draft_model_inputs, draft_len, draft_beam_size, beam_scores, beam_sequence, logits_processor, logits_warper)
        # 2. target
        target_outputs = target_beam_search(target_model, target_model_inputs, draft_outputs, draft_beam_size, beam_scores, beam_sequence, logits_processor)
        #3 verify
        verify_outputs = verify(target_model, target_model_inputs, draft_outputs, target_outputs, draft_beam_size, beam_size, beam_scores, beam_sequence, logits_processor, logits_warper)
        n_matches = verify_outputs["n_matches"]
        beam_sequence = verify_outputs["beam_sequence"]
        beam_scores = verify_outputs["beam_scores"]
        target_model_inputs = verify_outputs["target_model_inputs"]
        draft_model_inputs = verify_outputs["draft_model_inputs"]

        if accept_steps == []:
            one_step_beam_search_tc = draft_outputs["one_step_beam_search time_cost"]
            draft_forward_tc = draft_outputs["draft_forward time_cost"]
            logits_processor_tc = draft_outputs["logits_processor time_cost"]
            logits_processor_tc_per_beam = draft_outputs["logits_processor time_cost (per beam)"]
            target_forward_tc = target_outputs["target_forward time_cost"]
            verify_logits_processor_tc = verify_outputs["loop process beam_scores time_cost"]
            verify_logits_processor_tc_per_step = verify_outputs["logits_processor time_cost (per step)"]
            step1_draft_tc = draft_outputs["time_cost"]
            step1_target_tc = target_outputs["time_cost"]
            step1_verify_tc = verify_outputs["time_cost"]

        cur_len += n_matches + 1
        accept_rates.append(n_matches / draft_len)
        accept_steps.append(n_matches)
        draft_time_cost += draft_outputs['time_cost']
        target_time_cost += target_outputs['time_cost']
        verify_time_cost += verify_outputs['time_cost']
        lp_time_cost += verify_outputs['loop process beam_scores time_cost']
    n_run = len(accept_rates)
    ave_accept_rate = sum(accept_rates) / n_run
    total_accept_steps = sum(accept_steps)
    if target_model.generation_config.do_sample:  # Sort when do_sample
        beam_scores, sorted_indices = beam_scores.sort(descending=True)
        beam_sequence = beam_sequence[sorted_indices]
    return {
        "beam_sequence": beam_sequence,  # [beam_size, cur_len]
        "beam_scores": beam_scores,
        "n_run": n_run,
        "ave_accept_rate": ave_accept_rate,
        "total_accept_steps": total_accept_steps,
        "total_accept_tokens": total_accept_steps * beam_size,
        "ave_accept_tokens": total_accept_steps * beam_size / n_run,
        "draft_time_cost": draft_time_cost,
        "target_time_cost": target_time_cost,
        "verify_time_cost": verify_time_cost,
        'loop process beam_scores time_cost': lp_time_cost,
        'one_step_beam_search_tc': one_step_beam_search_tc,
        'draft_forward_tc': draft_forward_tc,
        'logits_processor_tc': logits_processor_tc,
        'logits_processor_tc_per_beam': logits_processor_tc_per_beam,
        'target_forward_tc': target_forward_tc,
        'verify_logits_processor_tc': verify_logits_processor_tc,
        'verify_logits_processor_tc_per_step': verify_logits_processor_tc_per_step,
        'step1_draft_tc': step1_draft_tc,
        'step1_target_tc': step1_target_tc,
        'step1_verify_tc': step1_verify_tc,    
    }

@Timer()
def target_generate(
    model, 
    inputs: Dict, 
    max_new_tokens: int, 
    logits_processor: Optional[LogitsProcessorList] = None,
    prefix_allowed_tokens_fn = None,
) -> Dict:
    # initialize
    beam_size = model.generation_config.num_beams
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    logits_processor = model._get_logits_processor(
        generation_config=model.generation_config,
        input_ids_seq_length=inputs["input_ids"].shape[-1],
        encoder_input_ids=inputs["input_ids"],
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs["input_ids"].device,
    )
    logits_warper = (
        model._get_logits_warper(model.generation_config) if model.generation_config.do_sample else None
    )
    n = inputs["input_ids"].shape[-1]
    min_dtype = torch.finfo(model.dtype).min
    causal_mask = (torch.tril(torch.ones((n, n), device=model.device)) == 0) * min_dtype
    inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": causal_mask[None, None, :, :],
        "position_ids": torch.arange(n, device=model.device).unsqueeze(0),
        "past_key_values": None
    }
    # Since logits_processor internally keeps track of beam_size, the input beam_scores and beam_sequence must be in batch form
    beam_scores = torch.zeros(inputs["input_ids"].shape[0], dtype=model.dtype, device=model.device)
    beam_sequence = inputs["input_ids"].repeat(beam_size, 1)
    # Auto-regressive generation
    for i in range(max_new_tokens):
        outputs = one_step_beam_search(model, inputs, beam_size, beam_scores, beam_sequence, logits_processor, logits_warper)
        inputs = {
            "input_ids": outputs["beam_tokens"][None, :],
            "attention_mask": outputs["attention_mask"],
            "position_ids": outputs["position_ids"],
            "past_key_values": outputs["past_key_values"]
        }
        beam_scores = outputs["beam_scores"]
        beam_sequence = outputs["beam_sequence"]
    if model.generation_config.do_sample:  # Sort when do_sample
        beam_scores, sorted_indices = beam_scores.sort(descending=True)
        beam_sequence = beam_sequence[sorted_indices]
    return {
        "beam_sequence": beam_sequence,
        "beam_scores": beam_scores
    }

