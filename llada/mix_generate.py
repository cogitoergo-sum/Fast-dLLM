import math
import torch
import numpy as np
import torch.nn.functional as F
import os
from typing import Optional
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
from generate import *
DECAY_RATE = math.log(0.4) / 40.0




@ torch.no_grad()
def mixed_generate(model, embedding_layer: torch.nn.modules.sparse.Embedding, warmed_tokens, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None, drop_prob=0.5, remask = False):
    '''
    Args:
        model: Mask predictor.
        embedding_layer: The embedding layer of the model.
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
    x[:, :prompt.shape[1]] = prompt.clone() # the non-masked tokens are currently still the prompt only
    mask_ids = torch.full(warmed_tokens_dropped.shape, mask_id, dtype=torch.long).to(model.device)
    mask_embed = embedding_layer(mask_ids)
    warmed_tokens_embed = embedding_layer(warmed_tokens_dropped)
    lerp_embed = torch.lerp(warmed_tokens_embed, mask_embed, 0.6)
    x_embed = embedding_layer(x)
    x_embed[:, prompt.shape[1]:prompt.shape[1] + warmed_tokens_dropped.shape[1]] = lerp_embed
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
            if nfe == 1:
                logits = model(inputs_embeds=x_embed).logits
            else:
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



def main():
    device = 'cuda'
    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    prompt = "An office has 90 workers. 2/3rds of them are men and the rest are women. The company hires 10 new employees and 100% of them are women. What is the total percentage of women in the company now?"
    answer = "40"
    prompt2 = "Colin can skip at six times the speed that Brandon can. Brandon can skip at one-third the speed that Tony can. And Tony can skip at twice the speed that Bruce can. At what speed, in miles per hour, can Colin skip if Bruce skips at 1 mile per hour?"
    answer2 = "4"
    m = [{"role": "user", "content": prompt}, ]
    m2 = [{"role": "user", "content": prompt2}, ]
    prompt = tokenizer.apply_chat_template(m, tokenize=False)
    prompt2 = tokenizer.apply_chat_template(m2, tokenize=False)
    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    input_ids2 = tokenizer(prompt2)['input_ids']
    input_ids2 = torch.tensor(input_ids2).to(device).unsqueeze(0)
    embedding_layer = model.model.transformer['wte'] # acqure the embedding layer
    def display_generation(title: str, outputs, input_tensor, answer):
        print(f"========= {title} =========")
        print("NFE =", outputs[1])
        generated_str: str = tokenizer.batch_decode(outputs[0][:, input_tensor.shape[1]:], skip_special_tokens=True)[0]
        if answer not in generated_str:
            print("Testing error: Swapped token led to false result, anticipated answer: ", answer)
        print("Output:", generated_str, "\n")
        return generated_str
    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, threshold=0.9, temperature=0., remasking='low_confidence')    
    gen1 = display_generation("Generate", out, input_ids, answer)    
    warmed_id1 = tokenizer(gen1, add_special_tokens=False)['input_ids']
    warmed_id1 = torch.tensor(warmed_id1).to(device).unsqueeze(0)

    out2 = generate(model, input_ids2, steps=128, gen_length=128, block_length=32, threshold=0.9, temperature=0., remasking='low_confidence')    
    gen2 = display_generation("Generate", out2, input_ids2, answer2)    
    warmed_id2 = tokenizer(gen2, add_special_tokens=False)['input_ids']
    warmed_id2 = torch.tensor(warmed_id2).to(device).unsqueeze(0)


    print("=========== TEST 4 : SWAPPED Warmed Generation with Remasking on Prompt 1 ===========")
    warmed_id1, warmed_id2 = warmed_id2, warmed_id1
    for i in range(10):
        drop_prob = i/10
        out = mixed_generate(model, embedding_layer, warmed_id1, input_ids, steps=128, gen_length=128, block_length=32, drop_prob=drop_prob, threshold=0.9, temperature=0., remasking='low_confidence', remask=False)
        display_generation(f"Swapped warmed generation with {drop_prob} drop rate", out, input_ids, answer)
    print("=========== TEST 5 : SWAPPED Warmed Generation with Remasking on Prompt 2 ===========")
    for i in range(10):
        drop_prob = i/10
        out = mixed_generate(model, embedding_layer, warmed_id2, input_ids2, steps=128, gen_length=128, block_length=32, drop_prob=drop_prob, threshold=0.9, temperature=0., remasking='low_confidence', remask=False)
        display_generation(f"Swapped warmed generation with {drop_prob} drop rate", out, input_ids2, answer2)


    
if __name__ == "__main__":
    main()