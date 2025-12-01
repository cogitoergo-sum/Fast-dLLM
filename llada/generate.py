# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import math
import torch
import numpy as np
import torch.nn.functional as F
import os
from typing import Optional
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
DECAY_RATE = math.log(0.4) / 40.0


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None, remask = False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            remask_bias = 2.5 * math.exp(DECAY_RATE * i) - 2 if remask else -1.0
            remask_bias = -0.0375*i+0.5 if remask else -1.0
            remask_indices = sample_remask_indices(logits, x, candidate_mask=transfer_index, bias=remask_bias)
            if remask_indices.numel() > 0:
                x[remask_indices[:, 0], remask_indices[:, 1]] = mask_id
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe



@ torch.no_grad()
def wrapper_generate(model, prompt, warmed_strings, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None, remask = False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    print(warmed_strings)
    # return generate(model, prompt, steps, gen_length, block_length, temperature, remasking, mask_id, threshold, factor, remask)
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    return x, 0


@ torch.no_grad()
def warmed_generate(model, warmed_tokens, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None, drop_prob=0.5, remask = False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        warmed_tokens: A tensor of shape (1, n), where n is the number of warmed tokens.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    # output is still gen_length long, but the first warmed_tokens.shape[1] tokens are replaced with warmed_tokens.
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    warmed_tokens_dropped = dropout_warmed_tokens(warmed_tokens, mask_id, drop_prob) # replace 50% of the warmed tokens with the mask token
    x[:, :prompt.shape[1]] = prompt.clone()
    x[:, prompt.shape[1]:prompt.shape[1] + warmed_tokens_dropped.shape[1]] = warmed_tokens_dropped.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            remask_bias = 2.5 * math.exp(DECAY_RATE * i) - 2 if remask else -1.0
            remask_bias = -0.0375*i+0.5
            remask_indices = sample_remask_indices(logits, x, candidate_mask=transfer_index, bias=remask_bias)
            if remask_indices.numel() > 0:
                x[remask_indices[:, 0], remask_indices[:, 1]] = mask_id
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe


def dropout_warmed_tokens(warmed_tokens: torch.Tensor, mask_id: int = 126336, drop_prob: float = 0.5) -> torch.Tensor:
    """
    Randomly replacing warmed tokens with the mask token.

    Args:
        warmed_tokens: Tensor of shape (batch_size, num_warmed) holding the warmed tokens.
        mask_id: The [MASK] token id (default: 126336).
        drop_prob: Probability that any given warmed token is replaced with the mask token.

    Returns:
        Tensor with the same shape; replaced positions contain mask_id.
    """
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError("drop_prob must be in [0, 1].")

    if drop_prob == 0.0:
        return warmed_tokens

    # create Bernoulli mask; True where we drop (set back to mask)
    drop_mask = torch.rand_like(warmed_tokens, dtype=torch.float32) < drop_prob

    return torch.where(drop_mask, torch.full_like(warmed_tokens, mask_id), warmed_tokens)


def sample_remask_indices(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
    bias: float = 0.0,
) -> torch.Tensor:
    """
    Select positions to be re-masked based on confidence and random sampling.

    Args:
        logits: Tensor of shape (batch, seq, vocab).
        token_ids: Tensor of shape (batch, seq) containing current token ids.
        candidate_mask: Optional boolean tensor of shape (batch, seq) indicating positions eligible
            for re-masking. If ``None``, all positions are considered.
        bias: Optional value added to the random draw before comparison. Positive values make
            re-masking more likely; negative values make it less likely.

    Returns:
        Tensor of shape (num_positions, 2) with (batch_idx, seq_idx) pairs to re-mask.
    """
    probs = F.softmax(logits.to(torch.float64), dim=-1)
    confidences = torch.gather(probs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    random_draw = torch.rand_like(confidences)
    remask_mask = random_draw + bias > confidences
    if candidate_mask is not None:
        remask_mask = remask_mask & candidate_mask
    return torch.nonzero(remask_mask, as_tuple=False)

@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe


@ torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], None, factor)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "An office has 90 workers. 2/3rds of them are men and the rest are women. The company hires 10 new employees and 100% of them are women. What is the total percentage of women in the company now?"
    answer = "40"
    prompt2 = "Hilary is shucking corn from ears that grew on her farm. She gets four ears of corn per stalk, and she has 108 stalks growing. Half the ears of corn have 500 kernels of corn and the other half have 100 more. How many kernels of corn does Hilary have to shuck?"
    answer2 = "65"
    print("Prompt 1: \n", prompt)
    print("Prompt 2: \n", prompt2)
    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    m2 = [{"role": "user", "content":  prompt2}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    prompt2 = tokenizer.apply_chat_template(m2, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    input_ids2 = tokenizer(prompt2)['input_ids']
    input_ids2 = torch.tensor(input_ids2).to(device).unsqueeze(0)
    
    def display_generation(title: str, outputs, input_tensor, answer):
        print(f"========= {title} =========")
        print("NFE =", outputs[1])
        generated_str: str = tokenizer.batch_decode(outputs[0][:, input_tensor.shape[1]:], skip_special_tokens=True)[0]
        if answer not in generated_str:
            print("Testing error: Swapped token led to false result.")
        print("Output:", generated_str, "\n")
        return generated_str

    print("=========== TEST 1 : Basic Generation ===========")
    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, threshold=0.9, temperature=0., remasking='low_confidence')
    generated_string1 = display_generation("Initial generation with 0.9 threshold", out, input_ids, answer)
    warmed_id1 = tokenizer(generated_string1)['input_ids']
    warmed_id1 = torch.tensor(warmed_id1).to(device).unsqueeze(0)

    out = generate(model, input_ids2, steps=128, gen_length=128, block_length=32, threshold=0.9, temperature=0., remasking='low_confidence')
    generated_string2 = display_generation("Initial generation with 0.9 threshold", out, input_ids2, answer2)
    warmed_id2 = tokenizer(generated_string2)['input_ids']
    warmed_id2 = torch.tensor(warmed_id2).to(device).unsqueeze(0)

    print("=========== TEST 2 : Basic Generation with Remasking ===========")
    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, threshold=0.9, temperature=0., remasking='low_confidence', remask=True)
    generated_string1 = display_generation("Initial generation with 0.9 threshold", out, input_ids, answer)
    out = generate(model, input_ids2, steps=128, gen_length=128, block_length=32, threshold=0.9, temperature=0., remasking='low_confidence', remask=True)
    generated_string2 = display_generation("Initial generation with 0.9 threshold", out, input_ids2, answer2)

    print("=========== TEST 3 : Warmed Generation ===========")
    out = warmed_generate(model, warmed_id1, input_ids, steps=128, gen_length=128, block_length=32, drop_prob=1.0, threshold=0.9, temperature=0., remasking='low_confidence')
    display_generation("Warmed generation with 1.0 drop rate (all drop)", out, input_ids, answer)

    out = warmed_generate(model, warmed_id1, input_ids, steps=128, gen_length=128, block_length=32, drop_prob=0.5, threshold=0.9, temperature=0., remasking='low_confidence')
    display_generation("Warmed generation with 0.5 drop rate", out, input_ids, answer)

    out = warmed_generate(model, warmed_id1, input_ids, steps=128, gen_length=128, block_length=32, drop_prob=0.0, threshold=0.9, temperature=0., remasking='low_confidence')
    display_generation("Warmed generation with 0 drop rate (no drop)", out, input_ids, answer)

    print("=========== TEST 4 : SWAPPED Warmed Generation with Remasking on Prompt 1 ===========")
    warmed_id1, warmed_id2 = warmed_id2, warmed_id1
    for i in range(10):
        drop_prob = i/10
        out = warmed_generate(model, warmed_id1, input_ids, steps=128, gen_length=128, block_length=32, drop_prob=drop_prob, threshold=0.9, temperature=0., remasking='low_confidence', remask=True)
        display_generation(f"Swapped warmed generation with {drop_prob} drop rate", out, input_ids, answer)

    print("=========== TEST 4.1 : SWAPPED Warmed Generation with Remasking on Prompt 2 ===========")
    for i in range(10):
        drop_prob = i/10
        out = warmed_generate(model, warmed_id2, input_ids2, steps=128, gen_length=128, block_length=32, drop_prob=drop_prob, threshold=0.9, temperature=0., remasking='low_confidence', remask=True)
        display_generation(f"Swapped warmed generation with {drop_prob} drop rate", out, input_ids2, answer2)

if __name__ == '__main__':
    main()
