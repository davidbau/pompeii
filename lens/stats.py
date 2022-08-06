import json
import os
import re
from collections import defaultdict

import numpy
import torch
from baukit import nethook

from datasets import load_dataset
from baukit import Covariance, tally
from baukit import TokenizedDataset, move_to, flatten_masked_batch, length_collation

def get_embedding_cov(model, tokenizer):
    
    def get_ds():
        ds_name = 'wikitext'
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
        )
        try:
            maxlen = model.config.n_positions
        except:
            maxlen = 100 # Hack due to missing setting in GPT2-NeoX.
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)
    ds = get_ds()
    sample_size = 1000
    batch_size = 5
    filename= None
    batch_tokens = 100

    progress = lambda x, **k: x

    stat = Covariance()
    loader = tally(
        stat,
        ds,
        cache=filename,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0
    )
    embed_layer = [n for n, _ in model.named_modules() if 'wte' in n or 'embed' in n][0]
    with torch.no_grad():
        for batch_group in loader:
            for batch in batch_group:
                move_to('cuda', batch)
                del batch['position_ids']
                with nethook.Trace(model, embed_layer) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                stat.add(feats.cpu().double())
    return stat.mean(), stat.covariance()

def make_generator_transform(mean=None, cov=None):
    d = len(mean) if mean is not None else len(cov)
    device = mean.device if mean is not None else cov.device
    layer = torch.nn.Linear(d, d, dtype=torch.double)
    nethook.set_requires_grad(False, layer)
    layer.to(device)
    layer.bias[...] = 0 if mean is None else mean
    if cov is None:
        layer.weight[...] = torch.eye(d).to(device)
    else:
        _, s, v = cov.svd()
        w = s.sqrt()[None, :] * v
        layer.weight[...] = w
    return layer

def collect_embedding_gaussian(model, tokenizer):
    m, c = get_embedding_cov(model, tokenizer)
    return make_generator_transform(m, c)

