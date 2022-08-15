
import os
from pathlib import Path

from .rome.rome import ROMEHyperParams, apply_rome_to_model
from .rome.util import nethook


def rewrite(layers, token_idx, target, prompt, model, tokenizer, model_name):

    nethook.set_requires_grad(True, model)

    hyperparams_path = os.path.join(Path(__file__).parent.resolve(), "rome/hparams", "ROME", f"{model_name}.json")

    hparams = ROMEHyperParams.from_json(hyperparams_path)
    hparams.layers = layers

    request = {
        "prompt": prompt,
        "token_idx" :token_idx,
        "target": target
    }

    edited_model, orig_weights = apply_rome_to_model(
        model, 
        tokenizer, 
        request, 
        hparams, 
        copy=True
    )

    return edited_model
