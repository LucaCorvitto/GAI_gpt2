import random
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import os
import random

SEED = 69
random.seed(SEED)

def temperature_top_p_sampling(model, device, tokenizer, prompt, entry_length, temperature=0, top_p=0.8, filter_value=-float("Inf")):
    entry_finished = False
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    for i in range(entry_length):
        outputs = model(generated, labels=generated)
        loss, logits = outputs[:2]
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0) #Temperature sampling
        # logits is divided by temperature before feeding them into softmax and obtaining the 
        # sampling probabilities
        #low temperature -> more confidence
        #high temperature (>1) -> less confidence

        sorted_logits, sorted_indices = torch.sort(logits, descending=True) #sorted in descenting order
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs >= top_p #total sum above top_p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value

        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).to(device)
        
        generated = torch.cat((generated, next_token), dim=1)

        if next_token in tokenizer.encode("<|endoftext|>"):
            entry_finished = True

        if entry_finished:
            
            cpu_gen = generated.squeeze().cpu()
            output_list = list(cpu_gen.numpy())
            output_text = tokenizer.decode(output_list)
            break

        elif len(generated[0]) > entry_length:
            break

    if not entry_finished:
        cpu_gen = generated.squeeze().cpu()
        output_list = list(cpu_gen.numpy())
        output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 

    return output_text

def sample_and_rank(model, device, tokenizer, prompt, entry_length, temperature=0.88, num_samples=20):
    """
    Samples N candidate responses from the logits distribution and returns the one with the highest probability.
    """
    sample_list = []
    # First compute the N candidates through plain random temperature sampling
    for i in range(num_samples):
        curr_prob = 1
        entry_finished = False
        generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

        for j in range(entry_length):
            outputs = model(generated, labels=generated)
            loss, logits = outputs[:2]
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0) #Temperature sampling
            #find a way to extract the probability of this token
            prob_dist = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(prob_dist, num_samples=1).to(device)
            token_prob = prob_dist[0][next_token] #since the softmax has just one dimension
            curr_prob *= token_prob.item()

            generated = torch.cat((generated, next_token), dim=1)

            if next_token in tokenizer.encode("<|endoftext|>"):
                entry_finished = True

            if entry_finished:
                
                cpu_gen = generated.squeeze().cpu()
                output_list = list(cpu_gen.numpy())
                output_text = tokenizer.decode(output_list)
                sample_list.append((output_text, curr_prob)) #collect output and its overall probability
                break

            elif len(generated[0]) > entry_length:
                break
        
        if not entry_finished:
            cpu_gen = generated.squeeze().cpu()
            output_list = list(cpu_gen.numpy())
            output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
            sample_list.append((output_text, curr_prob)) #collect output and its overall probability

    best = max(sample_list,key=lambda item:item[1])[0] 

    return best

def top_p_sample_and_rank(model, device, tokenizer, prompt, entry_length, temperature=0.88, num_samples=20, top_p=0.8, filter_value=-float("Inf")): #if 
    """
    Samples N candidate responses from the logits distribution using Top-p Sample and returns the one with the highest probability.
    """
    sample_list = []
    # First compute the N candidates through plain random temperature sampling
    for i in range(num_samples):
        curr_prob = 1
        entry_finished = False
        generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

        for j in range(entry_length):
            outputs = model(generated, labels=generated)
            loss, logits = outputs[:2]
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0) #Temperature sampling

            sorted_logits, sorted_indices = torch.sort(logits, descending=True) #sorted in descenting order
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p #total sum above top_p

            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value

            prob_dist = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(prob_dist, num_samples=1).to(device)
            token_prob = prob_dist[0][next_token]
            curr_prob *= token_prob.item()

            generated = torch.cat((generated, next_token), dim=1)

            if next_token in tokenizer.encode("<|endoftext|>"):
                entry_finished = True

            if entry_finished:
                
                cpu_gen = generated.squeeze().cpu()
                output_list = list(cpu_gen.numpy())
                output_text = tokenizer.decode(output_list)
                sample_list.append((output_text, curr_prob)) #collect output and its overall probability
                break

            elif len(generated[0]) > entry_length:
                break
        
        if not entry_finished:
            cpu_gen = generated.squeeze().cpu()
            output_list = list(cpu_gen.numpy())
            output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
            sample_list.append((output_text, curr_prob)) #collect output and its overall probability


    best = max(sample_list,key=lambda item:item[1])[0] 

    return best


def generate(
    model,
    tokenizer,
    prompt,
    device,
    top_p = None,
    num_samples = None,
    entry_count=10,
    entry_length=30, #maximum number of words
    temperature=1.,
):
    model.eval()
    model.to(device) #moved to device
    generated_list = []

    with torch.no_grad():

        if top_p and num_samples:
            print('Top_p_sample-and-rank')
            for entry_idx in trange(entry_count):
                to_add = top_p_sample_and_rank(model, device,tokenizer,prompt,entry_length,temperature,num_samples,top_p)
                generated_list.append(to_add)
        if top_p and not num_samples:
            print('Top_p_sample')
            for entry_idx in trange(entry_count):
                to_add = temperature_top_p_sampling(model, device,tokenizer,prompt,entry_length,temperature,top_p)
                generated_list.append(to_add)
        if num_samples and not top_p:
            print('Sample-and-rank')
            for entry_idx in trange(entry_count):
                to_add = sample_and_rank(model, device,tokenizer,prompt,entry_length,temperature,num_samples)
                generated_list.append(to_add)
        
        
                
    return generated_list

#code to clean the ouput generated in order to avoid uncomplete sentences
def clean_text(generated_text):
    cleaned_text = []
    for i in range(len(generated_text)):
        temp = generated_text[i]
        last_occ1 = temp.rfind('.')
        last_occ2 = temp.rfind('!')
        last_occ3 = temp.rfind('?')
        temp = temp[:max(last_occ1,last_occ2,last_occ3)+1]
        temp = temp.replace('\n\n', ' ')
        temp = temp.replace('\n', ' ')
        cleaned_text.append(temp)
    return cleaned_text