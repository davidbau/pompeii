from decimal import Overflow
import logging
import logging.config
import os
import sys
from functools import partial

import torch

import baukit
from . import logitlens
from .rewrite import rewrite


class LogitLensSession:

    def __init__(self, model_name):

        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': True
        })

        logging.basicConfig(filename='log.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

        self.model_name = model_name
        self.model, self.tokenizer = logitlens.get_model_tokenizer(model_name)
        self.decoder = torch.nn.Sequential(self.model.transformer.ln_f, self.model.lm_head)

        self.default_style = baukit.show.Style({'text-align' : 'center'}, color='black', display='block', overflow='hidden')
        self.bold_style = self.default_style + baukit.show.Style(fontWeight='bold')

        self.prompt = None
        self.prompt_tokens = None

        self.model_rewrite = None

        def hover(probabilities, words):
            lines = []
            for probability, word in zip(probabilities, words):
                lines.append(f'{word}: prob {probability:.2f}')
            return baukit.show.Attr(title='\n'.join(lines))

        self.hover = hover


        self.__init_layout__()
      
    def __init_layout__(self):

        self.text_input = baukit.Textbox()
        self.hidden_state_function_dropdown = baukit.Datalist(choices=["Layer", "Layer Delta", "MLP", "Attn"], value="Layer").on('change', self.hidden_state_change)
        self.run_button = baukit.Button('Run').on('click', self.run)
        self.logit_lens = baukit.Div()
        self.rewrite_button = baukit.Button('Start rewrite', style=baukit.show.Style(display='none')).on('click', self.rewrite_select_layers)
        rwi_style = baukit.show.Style({'font-size': '18px'}) + self.default_style
        self.rewrite_info = baukit.Div(style=rwi_style)

        baukit.show(
            [[self.text_input, self.run_button, self.hidden_state_function_dropdown]], 
            [[self.rewrite_button, self.rewrite_info]], 
            baukit.show.Style(overflow='scroll'),
            self.logit_lens
        )

        self.logit_lens_layout = None

    def select_layer(self, button, layers):

        layers.add(int(button.label))

        style = baukit.show.Style({
            'background-color' : '#FFFF00'
        })
        button.style += style

        button.off('click')
        button.on('click', partial(self.deselect_layer, button, layers))

    def deselect_layer(self, button, layers):

        layers.remove(int(button.label))

        style = baukit.show.Style({
            'background-color' : ''
        })

        button.style += style

        button.off('click')
        button.on('click', partial(self.select_layer, button, layers))

    def select_token(self, button, all_buttons, token):

        token.text = button.label
        token.idx = int(button.attrs['token_idx'])

        selected_style = baukit.show.Style({
            'background-color' : '#FFFF00'
        })
        deselected_style = baukit.show.Style({
            'background-color' : ''
        })

        for _button in all_buttons:
            _button.style += deselected_style

        button.style += selected_style

    def rewrite_select_layers(self):

        self.rewrite_info.innerHTML = 'Select Layers'

        layers = set()

        for row in self.logit_lens_layout[1]:
            div = row[0]
            button = baukit.Button(div.innerHTML, style=div.style)
            button.on('click', partial(self.select_layer, button, layers))
            row[0] = button

        self.rewrite_button.label = 'Finish Selection'
        self.rewrite_button.off('click')
        self.rewrite_button.on('click', partial(self.rewrite_select_token, layers))

        self.display()

    def rewrite_select_token(self, layers):

        self.rewrite_info.innerHTML = 'Select Token'

        class Token:
            def __init__(self):
                self.text = ''
                self.idx = -1

        token = Token()

        for i, div in enumerate(self.logit_lens_layout[0][0]):
            if i == 0:
                continue
            button = baukit.Button(div.innerHTML, style=div.style, attrs=div.attrs)
            self.logit_lens_layout[0][0][i] = button

        all_buttons = self.logit_lens_layout[0][0][1:]
        
        for button in all_buttons:
            button.on('click', partial(self.select_token, button, all_buttons, token))

        for row in self.logit_lens_layout[1]:
            button = row[0]
            div = baukit.Div(button.label, style=button.style)
            row[0] = div

        self.rewrite_button.off('click')
        self.rewrite_button.on('click', partial(self.rewrite_select_location, layers, token))
        
        self.display()

    def rewrite_select_location(self, layers, token):

        self.rewrite_info.innerHTML = 'Select ROME Location'
        
        for i, button in enumerate(self.logit_lens_layout[0][0]):
            if i == 0:
                continue
            div = baukit.Div(button.label, style=button.style)
            self.logit_lens_layout[0][0][i] = div

        for i, row in enumerate(self.logit_lens_layout[1]):
            for ii, div in enumerate(row):
                if ii == 0:
                    continue

                button = baukit.Button(div.innerHTML, style=div.style, attrs=div.attrs)
                button.on('click', partial(self.rewrite_select_new_token, layers, token, i, ii))
                
                row[ii] = button
        
        self.rewrite_button.style += baukit.show.Style(display='none')
        
        self.display()
    
    def rewrite_select_new_token(self, layers, token, row_idx, column_idx):
        
        self.rewrite_info.innerHTML = 'Input New Token'

        rome_location = (row_idx, column_idx - 1)

        for i, row in enumerate(self.logit_lens_layout[1]):
            for ii, button in enumerate(row):
                if ii == 0:
                    continue
                if i == row_idx and ii == column_idx:
                    token_input = baukit.Textbox(button.label, style=div.style, attrs=div.attrs)
                    token_input.on('enter', partial(self.rewrite, token_input, layers, token, rome_location))

                    row[ii] = token_input

                    continue

                div = baukit.Div(button.label, style=button.style, attrs=button.attrs)
                
                row[ii] = div

        self.rewrite_button.off('click')
        self.rewrite_button.on('click', partial(self.rewrite, token_input, layers, token, rome_location))
        self.rewrite_button.style += baukit.show.Style(display='block')
        self.rewrite_button.label = 'Rewrite'

        self.display()

    def rewrite(self, token_input, layers, token, rome_location):

        self.logit_lens.innerHTML = baukit.show.html(*self.logit_lens_layout)

        self.rewrite_button.style += baukit.show.Style(display='none')
        self.rewrite_info.innerHTML = 'Rewriting...'

        rome_layer_idx, rome_token_idx = rome_location

        self.logit_lens_layout[1][rome_layer_idx][rome_token_idx+1] = baukit.Div(token_input.value, style=self.default_style + self.color_fn(1.0), attrs=token_input.attrs)

        self.model_rewrite = rewrite(layers, token.idx, token_input.value, self.prompt, self.model, self.tokenizer, self.model_name)
        
        hidden_states, top_probs, top_words = self.process(self.model_rewrite)

        self.display_logit_lens_rewrite(top_probs, top_words)

        self.display()

    def display_logit_lens_rewrite(self, top_probs, top_words):

        for i, row in enumerate(self.logit_lens_layout[1]):

            _row = [row[0]]

            for ii, div in enumerate(row):
                if ii == 0:
                    continue

                style = baukit.show.Style({'border-bottom' : '4px solid #FFFF00'}) + self.default_style + self.color_fn(top_probs[i][ii-1][0])

                div = baukit.Div(top_words[i][ii-1][0], style=style, attrs=self.hover(top_probs[i][ii-1], top_words[i][ii-1]) + baukit.show.Attr(layer_idx=i, token_idx=ii-1))

                _row.extend([row[ii], div])

            self.logit_lens_layout[1][i] = _row

        self.rewrite_info.innerHTML = ''

    def display_logit_lens(self, top_probs, top_words):

        # header line
        header_line = [ 
                [baukit.Div('Layer', style=self.bold_style)] + [baukit.Div(token, style=self.default_style + baukit.show.Style({'border-style' : 'solid'}), attrs=baukit.show.Attr(token_idx=token_idx)) for token_idx, token in enumerate(self.prompt_tokens)]
            ]

        logit_lens_layout = [header_line,
            # body
            [[baukit.Div(str(layer), style=self.bold_style + baukit.show.Style({'border-style' : 'solid'}))]
            for layer in range(top_probs.shape[0])],
            header_line
            ]

        self.logit_lens_layout = logit_lens_layout
        self.display_logit_lens_inner(top_probs, top_words)

    def display_logit_lens_inner(self, top_probs, top_words):

        for layer, probabilites, words in zip(range(top_probs.shape[0]), top_probs, top_words):

            cells = []
            for token_idx, (_probabilites, _words) in enumerate(zip(probabilites, words)):
                div = baukit.Div(_words[0], style=self.default_style + self.color_fn(_probabilites[0]), attrs=self.hover(_probabilites, _words) + baukit.show.Attr(layer_idx=layer, token_idx=token_idx))
                cells.append(div)

            self.logit_lens_layout[1][layer].extend(cells)

    def hidden_state_change(self):

        if self.hidden_state_function_dropdown.value == "Layer":
            self.color = [0, 0, 255]
            self.hidden_state_function = logitlens.get_hidden_state_layers
        elif self.hidden_state_function_dropdown.value == "Layer Delta":
            self.color = [255, 0, 255]
            self.hidden_state_function = logitlens.get_hidden_state_layer_deltas
        elif self.hidden_state_function_dropdown.value == "MLP":
            self.color = [0, 255, 0]
            self.hidden_state_function = logitlens.get_hidden_state_mlp
        elif self.hidden_state_function_dropdown.value == "Attn":
            self.color = [255, 0, 0]
            self.hidden_state_function = logitlens.get_hidden_state_attn

        def color_fn(p):
            a = [int(255 * (1-p) + c * p) for c in self.color]
            return baukit.show.style(background=f'rgb({a[0]}, {a[1]}, {a[2]})')

        self.color_fn = color_fn

        if self.prompt:
            self.update()
            
    def update(self):

        hidden_states, top_probs, top_words = self.process(self.model)

        self.display_logit_lens(top_probs, top_words)

        if self.model_rewrite:

            hidden_states, top_probs, top_words = self.process(self.model_rewrite)
            self.display_logit_lens_rewrite(top_probs, top_words)

        self.display()

    def run(self):

        self.model_rewrite = None

        self.rewrite_button.style = baukit.show.style(display='block')

        self.prompt = self.text_input.value
        self.prompt_tokens = [self.tokenizer.decode(token) for token in self.tokenizer.encode(self.prompt)]

        self.hidden_state_change()

    def process(self, model, topk=5):

        with torch.no_grad():
    
            hidden_states = self.hidden_state_function(model, self.tokenizer, self.prompt)

            predictions = self.decoder(hidden_states).cpu()

            hidden_states = hidden_states.cpu()

            prediction_probabilities = torch.nn.functional.softmax(predictions, dim=-1)

            top_probs, top_tokens = prediction_probabilities.topk(k=topk, dim=-1)
            top_probs, top_tokens = top_probs[:,0], top_tokens[:,0]

            top_words = [[[self.tokenizer.decode(token) for token in _tokens] for _tokens in tokens] for tokens in top_tokens]

        return hidden_states, top_probs, top_words

    def display(self):
        self.logit_lens.innerHTML = baukit.show.html(*self.logit_lens_layout)


def generate(model, tokenizer, text, num_generation=10):

    input = {k: torch.tensor(v)[None].cuda() for k, v in tokenizer(text).items()}

    for _ in range(num_generation):

        output = model(**input)
        logits = output['logits']

        prediction = logits[0, -1].argmax()

        input['input_ids'] = torch.cat((input['input_ids'], torch.tensor([prediction])[None].cuda()), dim=1)
        input['attention_mask'] = torch.cat((input['attention_mask'], torch.ones(1, 1).cuda()), dim=1)
    
    return tokenizer.decode(input['input_ids'][0, len(input['input_ids']):])

