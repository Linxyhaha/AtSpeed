import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss

from transformers import LlamaForCausalLM
from transformers import GenerationConfig
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

import os
import random
import ipdb
from tqdm import tqdm



class AtSpeedSModel(LlamaForCausalLM):
    def __init__(self, config, teacher_model=None, prefix_allowed_tokens=None, alpha=0.5, topK=20, beta=1.0, r_original=1.0, r_student=0.0, generate_batchsize=4, teacher_batchsize=4, gamma=4, use_cache=True, world_size=2, debug=False, allowed_tokens_dict=None, constrained_softmax=False, temperature_softmax=1):
        super().__init__(config)
        self.alpha = alpha
        self.original_ratio = r_original
        self.student_ratio = r_student
        self.loss_CE_sum = CrossEntropyLoss(reduction="sum")
        self.loss_fct_KLD = KLDivLoss(reduction="batchmean", log_target=True)
        self.allowed_tokens_dict = allowed_tokens_dict
        self.constrained_softmax = constrained_softmax
        self.temperature_softmax = temperature_softmax

        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.freeze_model(self.teacher_model)

        self.prefix_allowed_tokens = prefix_allowed_tokens
        self.gamma = gamma
        self.topK = topK

        self.generate_batchsize = generate_batchsize
        self.teacher_batchsize = teacher_batchsize
        self.use_cache = use_cache
        self.debug = debug
        self.flag = 1

        self.world_size = world_size
        
        # if self.debug:        
        #     if int(os.environ.get("LOCAL_RANK") or 0) == 0:
        #         ipdb.set_trace()

    def freeze_model(self, model):

        for name, params in model.base_model.model.named_parameters():
            params.requires_grad = False

    def ensure_teacher_model_device(self):
        # fix the probable bug from the ddp
        self.teacher_model = self.teacher_model.to(f"cuda:{int(os.environ.get('LOCAL_RANK')) + self.world_size}")
        print(f"teacher model is on device {self.teacher_model.device}")
        
    def loss_strict(self, q_logits, p_logits, K, constrained=True):
        assert q_logits.shape == p_logits.shape, f"q_logits and p_logits should be of the same shape!! {q_logits.shape} != {p_logits.shape}"
        
        if constrained:
            if q_logits.shape[-1] == p_logits.shape[-1] and q_logits.shape[-1] > 32000:
                q_logits, p_logits = q_logits[..., 32000:], p_logits[..., 32000:]
        
        if self.allowed_tokens_dict is not None:  # set logits = -inf for not allowed tokens to mask probs
            if self.flag:
                self.vocab = torch.arange(q_logits.shape[-1]).to(q_logits.device)
                self.allowed_tokens_dict = {k: torch.tensor(list(v)).to(q_logits.device) - 32000 for k, v in tuple(self.allowed_tokens_dict.items())[:self.gamma]}
                self.not_allowed_tokens_bool_dict = {k: ~torch.isin(self.vocab, v) for k, v in self.allowed_tokens_dict.items()}
                self.flag = 0
            for k, v in self.allowed_tokens_dict.items():
                not_allowed_tokens = self.not_allowed_tokens_bool_dict[k]
                p_logits[:, k, not_allowed_tokens] = -float("inf")
                q_logits[:, k, not_allowed_tokens] = -float("inf")

        p_log_probs = F.log_softmax(p_logits, dim=-1) # (batch_size, seq_len, K)
        q_log_probs = F.log_softmax(q_logits, dim=-1) # (batch_size, seq_len, K)

        q_log_probs_topk, q_topk_index = torch.topk(q_log_probs, K)
        p_log_probs_topk = torch.gather(p_log_probs, -1, q_topk_index) # (batch_size, seq_len, K)

        if self.constrained_softmax:
            p_logits_topk = torch.exp(p_log_probs_topk)
            q_logits_topk = torch.exp(q_log_probs_topk)
             
            q_logits_topk = q_logits_topk / self.temperature_softmax
            p_logits_topk = p_logits_topk / self.temperature_softmax
            
            p_log_probs = F.log_softmax(p_logits_topk, dim=-1)
            q_log_probs = F.log_softmax(q_logits_topk, dim=-1)
        else:
            p_log_probs = p_log_probs_topk
            q_log_probs = q_log_probs_topk
        
        # set nan to 0 both for loss calculation
        p_log_probs = p_log_probs.nan_to_num(0)
        q_log_probs = q_log_probs.nan_to_num(0)

        p_log_probs = p_log_probs.view(-1, K).clone()
        q_log_probs = q_log_probs.view(-1, K).clone()

        loss_rKLD = self.loss_fct_KLD(p_log_probs, q_log_probs)

        p_probs_k = torch.exp(p_log_probs[..., -1]) # the k-th prob of the teacher model, (batch_size*seq_len, 1)
        loss_reg = -torch.mean(p_probs_k * torch.sum((torch.exp(q_log_probs) * q_log_probs), dim=-1)) 

        loss = loss_rKLD + loss_reg
        
        return loss


    def student_generate(self, batch_size, beam_size, past_kv=None, **data):
        '''
            Description: Given prefix, student need to autoregressively generates the output.
            Input:
                ``batch_size``: autoregrssive generation batch size
                ``beam_size``: generation beam size
                ``past_kv``: past key values of the student prefix for student generation
                ``**data``: input dictionary contains input_ids, attention_mask, position_ids, etc
            Return:
                ``outputs``: The beam sequences generated by student model.
        '''

        def batch(data, batch_size=batch_size):
            if isinstance(data, dict):
                data_len = len(data['input_ids']) 
                chunk_size = (data_len - 1) // batch_size + 1
                for i in range(chunk_size):
                    yield {k:v[batch_size * i: batch_size * (i + 1)] for k,v in data.items()} 
            else:
                data_len = len(data)
                chunk_size = (data_len - 1) // batch_size + 1
                for i in range(chunk_size):
                    yield data[batch_size * i: batch_size * (i + 1)] 
        
        def get_output(beam_size, past_kv, **inputs):
            inputs["input_ids"] = inputs["input_ids"][...,:-self.gamma]
            inputs["attention_mask"] = inputs["attention_mask"][...,:-self.gamma]
            dim = past_kv[0][0].size()
            past_kv = ((past_kv[0][0][:,:,:-self.gamma-1,:].repeat(1,self.topK,1,1).view(inputs["input_ids"].shape[0]*self.topK, dim[1], dim[2]-self.gamma-1, dim[3]), 
                        past_kv[0][1][:,:,:-self.gamma-1,:].repeat(1,self.topK,1,1).view(inputs["input_ids"].shape[0]*self.topK, dim[1], dim[2]-self.gamma-1, dim[3])),
                        (past_kv[1][0][:,:,:-self.gamma-1,:].repeat(1,self.topK,1,1).view(inputs["input_ids"].shape[0]*self.topK, dim[1], dim[2]-self.gamma-1, dim[3]),
                        past_kv[1][1][:,:,:-self.gamma-1,:].repeat(1,self.topK,1,1).view(inputs["input_ids"].shape[0]*self.topK, dim[1], dim[2]-self.gamma-1, dim[3])))

            output = self.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                past_key_values=past_kv,
                max_new_tokens=10,
                prefix_allowed_tokens_fn=self.prefix_allowed_tokens,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
                generation_config=generation_config
            )
            return output
        
        generation_config = GenerationConfig(
            temperature=1,
            do_sample=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            output_hidden_states=False,
            output_attentions=False)

        self.eval()
        outputs = []
        with torch.no_grad():
            self._forward = self.forward
            self.forward = self.forward_generate
            for i, (inputs, k0, v0, k1, v1) in tqdm(enumerate(zip(batch(data), batch(past_kv[0][0]), batch(past_kv[0][1]), batch(past_kv[1][0]), batch(past_kv[1][1])))):
                kv = ((k0, v0), (k1, v1))
                output = get_output(beam_size, kv, **inputs)
                output_ids = output["sequences"][...,-self.gamma:] # TIGER/LC-Rec
                output_ids = output_ids.view(len(k0), beam_size, -1)
                outputs = outputs + [output_ids]

            # (batch size, topk, gamma)
            outputs = torch.cat(outputs, dim=0)
            self.forward = self._forward

        self.train()
        return outputs
    
    def get_teacher_logits(self, data):
        '''
            input:
                ``data``: the data that contains input_ids, attention_mask, position_ids, etc. (the input that the model forward needs)
            return:
                ``output``: logits of every position (-5:-1).
        '''
        # 1. input the data
        # 2. obtain the logits of every position (-5:-1)

        assert self.teacher_model is not None, "Teacher model is None!"

        data = {k:v.to(f"cuda:{int(os.environ.get('LOCAL_RANK'))+self.world_size}") for k,v in data.items()}
        
        def batch(data, batch_size=self.teacher_batchsize):
            if isinstance(data, dict):
                data_len = len(data['input_ids']) 
                chunk_size = (data_len - 1) // batch_size + 1
                for i in range(chunk_size):
                    yield {k:v[batch_size * i: batch_size * (i + 1)] for k,v in data.items()} 

        outputs = []
        for idx, batch_data in tqdm(enumerate(batch(data))):
            # if int(os.environ.get("LOCAL_RANK") or 0) == 0:
            #     ipdb.set_trace()
            # dist.barrier()  # check that each item of data is a tensor and corresponds to the sequence input length

            output = self.teacher_model(**batch_data)
            output = output[..., -self.gamma-1:-1] 
            outputs = outputs + [output]

        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.to(f"cuda:{int(os.environ.get('LOCAL_RANK'))}")

        return output

    def combine_x_y(self, data, y):
        # 1. data is the original sample, y is the model-generated output.
        # 2. this function needs to return the combined sample with data[:-5] and y concatenating together

        input_ids = data['input_ids'].repeat(1, self.topK).view(-1, data['input_ids'].shape[-1])
        attention_mask = data['attention_mask'].repeat(1, self.topK).view(-1, data['attention_mask'].shape[-1])

        input_ids = torch.cat([input_ids[..., :-self.gamma], y.view(input_ids.shape[0],-1)], dim=-1)
        
        update_data = {"input_ids":input_ids, "attention_mask":attention_mask}

        return update_data

    def select_data(self, data, past_key_values=None):
        # 1. random sample u1, u2 to decide whether to use the original data or whether to use student model-generated data in model-generated data
        # 2. after the sampling decision process, returns the input dict
        u1 = random.random()
        if u1 < self.original_ratio:
            return ("original", data)
        else:
            u2 = random.random()
            if u2 < self.student_ratio:
                y_student = self.student_generate(self.generate_batchsize, self.topK, past_key_values, **data)
                # sample = self.combine_x_y(data, y_student)
                data = {k:v for k,v in data.items() if k!="teacher_output"}  # remove teacher output for combine
                sample = self.combine_x_y(data, y_student)
                sample_logits = self.get_teacher_logits(sample)
                return ("student", (sample, sample_logits))
            else:
                return ("teacher", data)
            
    def forward_generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        teacher_logits: Optional[torch.FloatTensor] = None,
        teacher_output: Optional[torch.LongTensor] = None,
        teacher_output_logits: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        teacher_logits: Optional[torch.FloatTensor] = None,
        teacher_output: Optional[torch.LongTensor] = None,
        teacher_output_logits: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        teacher_Yp_logits = teacher_output_logits 

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float() # student logits with original input


        # formulate for data selection
        data = {"input_ids":input_ids, 
                "attention_mask": attention_mask,
                "teacher_output": teacher_output}
        
        # choose using original data, teacher-generated data, or student-generated data
        inputs = self.select_data(data, outputs.past_key_values)

        source, inputs = inputs[0], inputs[1]
        
        if len(inputs) == 2: # if the student-generated data is chosen, it will return data and teacher logits, two elements
            inputs, teacher_logits = inputs[0], inputs[1]

        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        
        # if int(os.environ.get("LOCAL_RANK") or 0) == 0:
        #     ipdb.set_trace()
        # dist.barrier()


        # DESCRIPTION: If use model-generated data, the forward can reuse kv_cache
        # 1. use the original data for forward, get the original student prediction (SFT)
        # 2. use kv-cache to go for model-generated y, and get the student prediction
        # 3. calculate KD loss with student prediction and teacher logits
        # 4. (1-alpha) * original loss + alpha * KD_loss

        # if original, calculate loss
        if source == "original":
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                ### Added lines
                vocab_size = teacher_logits.shape[-1]
                student_logits = logits[:, -self.gamma-1:-1].contiguous()
                loss_KLD = self.loss_strict(student_logits, teacher_logits, self.topK)
                loss = (1 - self.alpha) * loss + self.alpha * loss_KLD
                ### End of added lines

        elif source == "teacher":
            if self.use_cache:
                # if self.debug:
                #     print("Now is DEBUG mode!!!")
                #     teacher_output = torch.ones((input_ids.shape[0], self.topK, self.gamma), dtype=torch.int64).cuda() # test ONLY!!!
                #     teacher_Yp_logits = torch.ones((input_ids.shape[0], self.topK, self.gamma, 33014), dtype=torch.float64).cuda()
                input_ids_X_Yp = teacher_output.view(-1, self.gamma) # (bs*topk, gamma)
                attention_mask_X_Yp = attention_mask.repeat(1, self.topK).view(-1, attention_mask.shape[-1])
                position_ids_X_Yp = position_ids if position_ids is None else position_ids.repeat(1, self.topK).view(-1, self.gamma)
                kv_tuple = outputs.past_key_values
                dim = kv_tuple[0][0].size()
                past_key_values = ((kv_tuple[0][0][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yp.shape[0], dim[1], dim[2]-self.gamma, dim[3]), 
                                    kv_tuple[0][1][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yp.shape[0], dim[1], dim[2]-self.gamma, dim[3])),
                                    (kv_tuple[1][0][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yp.shape[0], dim[1], dim[2]-self.gamma, dim[3]),
                                    kv_tuple[1][1][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yp.shape[0], dim[1], dim[2]-self.gamma, dim[3])))
                
                outputs_X_Yp = self.model(
                    input_ids=input_ids_X_Yp,
                    attention_mask=attention_mask_X_Yp,
                    position_ids=position_ids_X_Yp,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True, 
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )

                hidden_states = outputs_X_Yp[0]
                if self.config.pretraining_tp > 1:
                    lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                    student_Yp_logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                    student_Yp_logits = torch.cat(student_Yp_logits, dim=-1)
                else:
                    student_Yp_logits = self.lm_head(hidden_states)
                student_Yp_logits = student_Yp_logits.float()
                student_Yp_logits = torch.cat([logits[:,-self.gamma-1,:].unsqueeze(1).repeat(1,self.topK,1).view(-1,1,logits.shape[-1]), student_Yp_logits[:,-self.gamma:-1,:]], dim=1)  # shift to get the logits

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

                # 1. student_Yp_logits is the student logits of the teacher output
                # 2. teacher_Yp_logits is pre-saved, can directly use, but ensure the logits lengths are the same as student_Yp_logits
                student_Yp_logits = student_Yp_logits[...,32000:]
                student_Yp_logits = student_Yp_logits.contiguous()

                teacher_Yp_logits = teacher_Yp_logits.view(-1, student_Yp_logits.shape[-2], student_Yp_logits.shape[-1])
                teacher_Yp_logits = teacher_Yp_logits.to(student_Yp_logits.device) # 存的是 -gamma-1:-1
                
                loss_KD = self.loss_strict(student_Yp_logits, teacher_Yp_logits, self.topK)    

                loss = (1 - self.alpha) * loss + self.alpha * loss_KD

        elif source == "student":
            teacher_Yq_logits = teacher_logits

            if self.use_cache:
                if self.debug:
                    print("Now is DEBUG mode!!!")
                    student_output = torch.ones((input_ids.shape[0], self.topK, self.gamma), dtype=torch.int64).cuda() # test ONLY!!!
                    teacher_Yq_logits = torch.ones((input_ids.shape[0], self.topK, self.gamma, 1014), dtype=torch.float64).cuda()
                input_ids_X_Yq = student_output.view(-1, self.gamma) # (bs*topk, gamma)
                attention_mask_X_Yq = attention_mask.repeat(1, self.topK).view(-1, attention_mask.shape[-1])
                position_ids_X_Yq = position_ids if position_ids is None else position_ids.repeat(1, self.topK).view(-1, self.gamma)
                kv_tuple = outputs.past_key_values
                dim = kv_tuple[0][0].size()
                past_key_values = ((kv_tuple[0][0][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yq.shape[0], dim[1], dim[2]-self.gamma, dim[3]), 
                                    kv_tuple[0][1][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yq.shape[0], dim[1], dim[2]-self.gamma, dim[3])),
                                    (kv_tuple[1][0][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yq.shape[0], dim[1], dim[2]-self.gamma, dim[3]),
                                     kv_tuple[1][1][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yq.shape[0], dim[1], dim[2]-self.gamma, dim[3])))
                
                outputs_X_Yq = self.model(
                    input_ids=input_ids_X_Yq,
                    attention_mask=attention_mask_X_Yq,
                    position_ids=position_ids_X_Yq,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True, 
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )

                hidden_states = outputs_X_Yq[0]
                if self.config.pretraining_tp > 1:
                    lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                    student_Yq_logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                    student_Yq_logits = torch.cat(student_Yq_logits, dim=-1)
                else:
                    student_Yq_logits = self.lm_head(hidden_states)
                student_Yq_logits = student_Yq_logits.float()
                student_Yq_logits = torch.cat([logits[:,-self.gamma-1,:].unsqueeze(1).repeat(1,self.topK,1).view(-1,1,logits.shape[-1]), student_Yq_logits[:,-self.gamma:-1,:]], dim=1)  # shift to get the logits

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

                # 1. student_Yq_logits is the student logits of the student output
                # 2. teacher_Yq_logits is online calculated, should ensure the logits lengths are the same as student_Yq_logits
                vocab_size = student_Yq_logits.shape[-1]
                student_Yq_logits = student_Yq_logits.contiguous()

                teacher_Yp_logits = teacher_Yp_logits.view(-1, student_Yp_logits.shape[-2], student_Yp_logits.shape[-1])
                teacher_Yq_logits = teacher_Yq_logits.to(student_Yq_logits.device)  # [-gamma-1:-1]
                loss_KD = self.loss_strict(student_Yq_logits, teacher_Yq_logits, self.topK)    

                loss = (1 - self.alpha) * loss + self.alpha * loss_KD

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        outputs.past_key_values = None
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
  

class AtSpeedRModel(LlamaForCausalLM):
    def __init__(self, config, alpha=0.5, topK=20, gamma=5, use_cache=True, debug=False, allowed_tokens_dict=None, constrained_softmax=False, temperature_softmax=1):
        super().__init__(config)
        self.topK = topK
        self.gamma = gamma
        self.alpha = alpha
        self.loss_fct_KLD = KLDivLoss(reduction="batchmean", log_target=True)
        self.allowed_tokens_dict = allowed_tokens_dict
        self.constrained_softmax = constrained_softmax
        self.temperature_softmax = temperature_softmax

        self.use_cache = use_cache
        self.debug = debug
        self.flag = 1

    def loss_soft(self, q_logits, p_logits, constrained=True):
        assert q_logits.shape == p_logits.shape, "q_logits and p_logits should be of the same shape!!"

        if constrained and q_logits.shape[-1] > 32000:
            q_logits, p_logits = q_logits[..., 32000:], p_logits[..., 32000:]  # make an adjustment to the logits range, limiting it to the new vocabulary

        if self.allowed_tokens_dict is not None:  # set logits = -inf for not allowed tokens to mask probs
            if self.flag:
                self.vocab = torch.arange(q_logits.shape[-1]).to(q_logits.device)
                self.allowed_tokens_dict = {k: torch.tensor(list(v)).to(q_logits.device) - 32000 for k, v in tuple(self.allowed_tokens_dict.items())[:self.gamma]}
                self.not_allowed_tokens_bool_dict = {k: ~torch.isin(self.vocab, v) for k, v in self.allowed_tokens_dict.items()}
                self.flag = 0
            if self.constrained_softmax:
                for k, v in self.allowed_tokens_dict.items():
                    not_allowed_tokens = self.not_allowed_tokens_bool_dict[k]
                    p_logits[:, k, not_allowed_tokens] = -float("inf")
                    q_logits[:, k, not_allowed_tokens] = -float("inf")


        q_logits = q_logits / self.temperature_softmax
        p_logits = p_logits / self.temperature_softmax
        q_probs = F.softmax(q_logits, dim=-1)
        p_probs = F.softmax(p_logits, dim=-1)
        q_probs = q_probs.clone()
        p_probs = p_probs.clone()

        if self.allowed_tokens_dict is not None:  # set student = teacher for not allowed tokens to mask loss
            for k, v in self.allowed_tokens_dict.items():
                not_allowed_tokens = self.not_allowed_tokens_bool_dict[k]
                p_probs[:, k, not_allowed_tokens] = 0
                q_probs[:, k, not_allowed_tokens] = 0

        loss_TVD = (0.5 * torch.abs(q_probs - p_probs)).sum(dim=-1).mean()
        return loss_TVD

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        teacher_logits: Optional[torch.FloatTensor] = None,
        teacher_output: Optional[torch.LongTensor] = None,
        teacher_output_logits: Optional[torch.FloatTensor] = None, 
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        teacher_Yp_logits = teacher_output_logits
        if teacher_output.shape[-1] > self.gamma: # with EOS
            teacher_output = teacher_output[...,-self.gamma:] # (bs, 20, gamma)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True, 
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        if self.use_cache:
            if self.debug:
                print("Now is DEBUG mode!!!")
                teacher_output = torch.ones((input_ids.shape[0], self.topK, self.gamma), dtype=torch.int64).cuda() # test ONLY!!!
                teacher_Yp_logits = torch.ones((input_ids.shape[0], self.topK, self.gamma, 1014), dtype=torch.float64).cuda()
            input_ids_X_Yp = teacher_output.view(-1, self.gamma) # (bs*topk, gamma)
            attention_mask_X_Yp = attention_mask.repeat(1, self.topK).view(-1, attention_mask.shape[-1])
            position_ids_X_Yp = position_ids if position_ids is None else position_ids.repeat(1, self.topK).view(-1, self.gamma)
            kv_tuple = outputs.past_key_values
            dim = kv_tuple[0][0].size()
            past_key_values = ((kv_tuple[0][0][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yp.shape[0], dim[1], dim[2]-self.gamma, dim[3]), 
                                kv_tuple[0][1][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yp.shape[0], dim[1], dim[2]-self.gamma, dim[3])),
                                (kv_tuple[1][0][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yp.shape[0], dim[1], dim[2]-self.gamma, dim[3]),
                                 kv_tuple[1][1][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yp.shape[0], dim[1], dim[2]-self.gamma, dim[3])))
            
            outputs_X_Yp = self.model(
                input_ids=input_ids_X_Yp,
                attention_mask=attention_mask_X_Yp,
                position_ids=position_ids_X_Yp,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True, 
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs_X_Yp[0]
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                student_Yp_logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                student_Yp_logits = torch.cat(student_Yp_logits, dim=-1)
            else:
                student_Yp_logits = self.lm_head(hidden_states)
            student_Yp_logits = student_Yp_logits.float()
            student_Yp_logits = torch.cat([logits[:,-self.gamma-1,:].unsqueeze(1).repeat(1,self.topK,1).view(-1,1,logits.shape[-1]), student_Yp_logits[:,-self.gamma:-1,:]], dim=1)  # shift to get the logits
        else:
            if self.debug:
                print("Now is DEBUG mode!!!")
                teacher_output = torch.ones((input_ids.shape[0], self.topK, self.gamma), dtype=torch.int64).cuda() # test ONLY!!
            seq_len = input_ids.shape[-1] - self.gamma # input length

            input_ids_X_Yp = teacher_output.view(-1, self.gamma) # (bs*topk, gamma)
            input_ids_X_Yp = torch.cat([input_ids[..., :seq_len].repeat(1, self.topK).view(-1, seq_len), input_ids_X_Yp], dim=-1) # self.gamma =4 if use LC-Rec

            attention_mask_X_Yp = attention_mask.repeat(1, self.topK).view(-1, attention_mask.shape[-1])
            position_ids_X_Yp = position_ids if position_ids is None else position_ids.repeat(1, self.topK).view(-1, self.gamma)
            past_key_values = None
            
            outputs_X_Yp = self.model(
                input_ids=input_ids_X_Yp,
                attention_mask=attention_mask_X_Yp,
                position_ids=position_ids_X_Yp,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=False, 
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs_X_Yp[0][:,-self.gamma-1:-1,:]
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                student_Yp_logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                student_Yp_logits = torch.cat(student_Yp_logits, dim=-1)
            else:
                student_Yp_logits = self.lm_head(hidden_states)
            student_Yp_logits = student_Yp_logits.float()
            student_Yp_logits = torch.cat([logits[:,-self.gamma-1,:].unsqueeze(1).repeat(1,self.topK,1).view(-1,1,logits.shape[-1]), student_Yp_logits[:,-self.gamma:-1,:]], dim=1) # 前移一位拿logits

        # Compute loss
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            ### Added lines
            student_Yp_logits = student_Yp_logits[..., 32000:].contiguous() # (bs*topk, gamma, n_new_tokens)
            teacher_Yp_logits = teacher_Yp_logits.view(-1, self.gamma, teacher_Yp_logits.shape[-1]) # (bs*topk, gamma, n_new_tokens)
            # Enable model parallelism
            teacher_Yp_logits = teacher_Yp_logits.to(student_Yp_logits.device)
            loss_KLD = self.loss_soft(student_Yp_logits, teacher_Yp_logits)
            loss = (1 - self.alpha) * loss + self.alpha * loss_KLD
            ### End of added lines

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        outputs.past_key_values = None
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class WordKDModel(LlamaForCausalLM):
    def __init__(self, config, alpha=0.5):
        super().__init__(config)
        self.alpha = alpha
        self.loss_fct_KLD = KLDivLoss(reduction="batchmean", log_target=True)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        teacher_logits: Optional[torch.FloatTensor] = None,
        teacher_output: Optional[torch.LongTensor] = None,
        teacher_output_logits: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            if self.alpha != 0:
                vocab_size = teacher_logits.shape[-1]
                student_logits = logits[:, -6:-1].contiguous()#.view(-1, vocab_size)
                teacher_logits = teacher_logits#.view(-1, vocab_size)
                student_log_probs = F.log_softmax(student_logits, dim=-1).clone()
                teacher_log_probs = F.log_softmax(teacher_logits, dim=-1).clone()
                loss_KLD = self.loss_fct_KLD(student_log_probs.view(-1, vocab_size), teacher_log_probs.view(-1, vocab_size))
                loss = (1 - self.alpha) * loss + self.alpha * loss_KLD

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TVDKDModel(LlamaForCausalLM):
    def __init__(self, config, alpha=0.5):
        super().__init__(config)
        self.alpha = alpha
        self.loss_fct_KLD = KLDivLoss(reduction="batchmean")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        teacher_logits: Optional[torch.FloatTensor] = None,
        teacher_output: Optional[torch.LongTensor] = None,
        teacher_output_logits: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            vocab_size = teacher_logits.shape[-1]
            student_logits = logits[:, -6:-1].contiguous().view(-1, vocab_size)
            teacher_logits = teacher_logits.view(-1, vocab_size)
            student_probs = F.softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            loss_TVD = (0.5 * torch.abs(student_probs - teacher_probs)).sum(dim=-1).mean()
            loss = (1 - self.alpha) * loss + self.alpha * loss_TVD

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class SeqKDModel(LlamaForCausalLM):
    def __init__(self, config, alpha=0.5, topK=20, gamma=5, use_cache=True):
        super().__init__(config)
        self.alpha = alpha
        self.topK = topK
        self.gamma = gamma
        self.use_cache = use_cache

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        teacher_logits: Optional[torch.FloatTensor] = None,
        teacher_output: Optional[torch.LongTensor] = None,
        teacher_output_logits: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
    
        if teacher_output.shape[-1] > self.gamma: # with EOS
            teacher_output = teacher_output[..., -self.gamma-1:-1] # (bs, 20, gamma)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True, # When training draft model with SeqKD, use_cache should be True to save training time!!
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        if self.use_cache:
            input_ids_X_Yp = teacher_output.view(-1, self.gamma) # (bs*topk, gamma)
            attention_mask_X_Yp = attention_mask.repeat(1, self.topK).view(-1, attention_mask.shape[-1])
            position_ids_X_Yp = position_ids if position_ids is None else position_ids.repeat(1, self.topK).view(-1, self.gamma)
            kv_tuple = outputs.past_key_values
            dim = kv_tuple[0][0].size()
            past_key_values = ((kv_tuple[0][0][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yp.shape[0], dim[1], dim[2]-self.gamma, dim[3]), 
                                kv_tuple[0][1][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yp.shape[0], dim[1], dim[2]-self.gamma, dim[3])),
                                (kv_tuple[1][0][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yp.shape[0], dim[1], dim[2]-self.gamma, dim[3]),
                                 kv_tuple[1][1][:,:,:-self.gamma,:].repeat(1,self.topK,1,1).view(input_ids_X_Yp.shape[0], dim[1], dim[2]-self.gamma, dim[3]))) # NOTE: This is model-specific. Currently, it is compatible for LLama-68M, with 2 layer and 12 heads, with 768 hidden dimension.
            
            outputs_X_Yp = self.model(
                input_ids=input_ids_X_Yp,
                attention_mask=attention_mask_X_Yp,
                position_ids=position_ids_X_Yp,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True, # When training draft model with SeqKD, should set use_cache=True to speeup the training
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs_X_Yp[0]
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                student_Yp_logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                student_Yp_logits = torch.cat(student_Yp_logits, dim=-1)
            else:
                student_Yp_logits = self.lm_head(hidden_states)

            student_Yp_logits = student_Yp_logits.float()
            student_Yp_logits = torch.cat([logits[:,-self.gamma-1,:].unsqueeze(1).repeat(1,self.topK,1).view(-1,1,logits.shape[-1]), student_Yp_logits[:,-self.gamma:-1,:]], dim=1)  # shift to get the logits
        
        else:
            print("Warning: use_cache=False. This will lower the training speed :(((")
            seq_len = input_ids.shape[-1] - self.gamma # input length

            input_ids_X_Yp = teacher_output.view(-1, self.gamma) # (bs*topk, gamma)
            input_ids_X_Yp = torch.cat([input_ids[..., :seq_len].repeat(1, self.topK).view(-1, seq_len), input_ids_X_Yp], dim=-1) # self.gamma =4 if use LC-Rec

            attention_mask_X_Yp = attention_mask.repeat(1, self.topK).view(-1, attention_mask.shape[-1])
            position_ids_X_Yp = position_ids if position_ids is None else position_ids.repeat(1, self.topK).view(-1, self.gamma)
            past_key_values = None
            
            outputs_X_Yp = self.model(
                input_ids=input_ids_X_Yp,
                attention_mask=attention_mask_X_Yp,
                position_ids=position_ids_X_Yp,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs_X_Yp[0][:,-self.gamma-1:-1,:]
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                student_Yp_logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                student_Yp_logits = torch.cat(student_Yp_logits, dim=-1)
            else:
                student_Yp_logits = self.lm_head(hidden_states)

            student_Yp_logits = student_Yp_logits.float()
            student_Yp_logits = torch.cat([logits[:,-self.gamma-1,:].unsqueeze(1).repeat(1,self.topK,1).view(-1,1,logits.shape[-1]), student_Yp_logits[:,-self.gamma:-1,:]], dim=1)  # shift to get the logits

        loss = None
        if labels is not None:

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            # 1. student_Yp_logits is the student logits of the teacher output
            # 2. label should be label_repeat with topk [-gamma] + teacher_output[-gamma+1:]
            vocab_size = student_Yp_logits.shape[-1]
            student_Yp_logits = student_Yp_logits.contiguous()
            student_Yp_logits = student_Yp_logits.view(-1, vocab_size)

            teacher_labels = labels.repeat(1, self.topK).view(-1, labels.shape[-1])
            teacher_output = teacher_output.view(teacher_output.shape[0]*teacher_output.shape[1], -1)
            teacher_label = torch.cat([teacher_labels[...,-self.gamma].unsqueeze(-1), teacher_output[..., -self.gamma+1:]], dim=1)

            teacher_label = teacher_label.view(-1)

            teacher_label = teacher_label.to(student_Yp_logits.device)
            loss_KD = loss_fct(student_Yp_logits, teacher_label)

            loss = (1 - self.alpha) * loss + self.alpha * loss_KD

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

