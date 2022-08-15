import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import baukit
from .rome.util import nethook


def get_model_tokenizer(model_name):

    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.pad_token = tokenizer.eos_token

    nethook.set_requires_grad(False, model)

    return model, tokenizer


def get_hidden_states(model, tokenizer, prefix, layers=None, layer_key=None, other_keys=None):
    
    if not layers:
        layers = list(range(
            len([n for n, _ in model.named_modules()
             if re.match('^transformer.h.\d+$', n)])))

    layer_names = [f"transformer.h.{i}{'.' + layer_key if layer_key else ''}" for i in layers]

    if other_keys:
        layer_names = other_keys + layer_names

    input = tokenizer(prefix, return_tensors='pt')

    input = {key: value[None].cuda() for key, value in input.items()}

    with baukit.TraceDict(model, layer_names) as tr:
        model(**input)['logits']

    return tr

def get_hidden_state_layers(model, tokenizer, prefix, layers=None):

    tr = get_hidden_states(model, tokenizer, prefix, layers=layers)

    hidden_states = torch.stack([tr[layer_name].output[0] for layer_name in tr.keys()])

    return hidden_states

def get_hidden_state_mlp(model, tokenizer, prefix, layers=None):

    tr = get_hidden_states(model, tokenizer, prefix, layers=layers, layer_key='mlp')

    hidden_states = torch.stack([tr[layer_name].output[0] for layer_name in tr.keys()])

    return hidden_states[:, None, :, :]

def get_hidden_state_attn(model, tokenizer, prefix, layers=None):

    tr = get_hidden_states(model, tokenizer, prefix, layers=layers, layer_key='attn')

    hidden_states = torch.stack([tr[layer_name].output[0] for layer_name in tr.keys()])

    return hidden_states

def get_hidden_state_layer_deltas(model, tokenizer, prefix, layers=None):

    tr = get_hidden_states(model, tokenizer, prefix, layers=layers, other_keys=['transformer.drop'])

    hidden_states = torch.stack([tr[layer_name].output[0] for layer_name in tr.keys() if layer_name != 'transformer.drop'])
    first_hidden_state = tr['transformer.drop'].output[None]
    hidden_states = torch.cat([first_hidden_state, hidden_states])
    delta_hidden_states = hidden_states[1:] - hidden_states[:-1]

    return delta_hidden_states
