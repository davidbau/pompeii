
import os

from _rome.rome import ROMEHyperParams, apply_rome_to_model
from _rome.util import nethook

def rewrite(layers, token, target, prompt, model, tokenizer, model_name):

    copy = False

    nethook.set_requires_grad(True, model)

    hyperparams_path = os.path.join("rome/hparams", "ROME", f"{model_name}.json")

    hparams = ROMEHyperParams.from_json(hyperparams_path)
    hparams.layers = layers

    prompt = prompt.replace(token, '{}')

    if prompt[-1] != ' ':
        prompt += ' '

    print(prompt)

    request = {
        "prompt": prompt,
        "subject": token,
        "target_new": {
            "str": target
        }
    }

    edited_model, orig_weights = apply_rome_to_model(
        model, 
        tokenizer, 
        request, 
        hparams, 
        return_orig_weights=False
    )

    return edited_model