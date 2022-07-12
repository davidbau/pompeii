import os
import sys
from functools import partial

import torch

import baukit
import logitlens
from rewrite import rewrite


class LogitLensSession:

    def __init__(self, model_name):

        self.model_name = model_name
        self.model, self.tokenizer = logitlens.get_model_tokenizer(model_name)
        self.decoder = torch.nn.Sequential(self.model.transformer.ln_f, self.model.lm_head)

        self.default_style = baukit.show.Style(color='black')
        self.default_style.update({'text-align' : 'center'})
        self.bold_style = self.default_style + baukit.show.Style(fontWeight='bold')

        self.prompt = None
        self.prompt_tokens = None
        self.hidden_states = None
        self.predictions = None
        self.prediction_probabilities = None
        self.top_probs = None
        self.top_tokens = None
        self.top_words = None

        self.__init_layout__()
      
    def __init_layout__(self):

        self.text_input = baukit.Textbox()
        self.run_button = baukit.Button('Run').on('click', lambda : self.run_logit_lens(self.text_input.value))
        self.hidden_state_function_dropdown = baukit.Datalist(choices=["Layer", "Layer Delta", "MLP", "Attn"], value="Layer")
        self.logit_lens = baukit.Div()
        self.rewrite_button = baukit.Button('Start rewrite', style=baukit.show.Style(display='none')).on('click', self.rewrite_select_layers)
        self.rewrite_info = baukit.Div(style=self.default_style)

        baukit.show(
            [[self.text_input, self.hidden_state_function_dropdown, self.run_button]], 
            [[self.rewrite_button, self.rewrite_info]], 
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

        token.value = button.label

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
        self.logit_lens.innerHTML = baukit.show.html(*self.logit_lens_layout)

    def rewrite_select_token(self, layers):

        self.rewrite_info.innerHTML = 'Select Token'

        class Token:
            def __init__(self):
                self.value = ''

        token = Token()

        for i, div in enumerate(self.logit_lens_layout[0][0]):
            if i == 0:
                continue
            button = baukit.Button(div.innerHTML, style=div.style)
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
        
        self.logit_lens.innerHTML = baukit.show.html(*self.logit_lens_layout)

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
        
        self.logit_lens.innerHTML = baukit.show.html(*self.logit_lens_layout)
    
    def rewrite_select_new_token(self, layers, token, row_idx, column_idx):
        
        self.rewrite_info.innerHTML = 'Input New Token'

        for i, row in enumerate(self.logit_lens_layout[1]):
            for ii, button in enumerate(row):
                if ii == 0:
                    continue
                if i == row_idx and ii == column_idx:
                    token_input = baukit.Textbox(button.label, style=div.style, attrs=div.attrs)
                    token_input.on('enter', partial(self.rewrite, token_input, layers, token))

                    row[ii] = token_input

                    continue

                div = baukit.Div(button.label, style=button.style, attrs=button.attrs)
                
                row[ii] = div

        self.rewrite_button.off('click')
        self.rewrite_button.on('click', partial(self.rewrite, token_input, layers, token))
        self.rewrite_button.style += baukit.show.Style(display='block')
        self.rewrite_button.label = 'Rewrite'

        self.logit_lens.innerHTML = baukit.show.html(*self.logit_lens_layout)

    def rewrite(self, token_input, layers, token):

        self.logit_lens.innerHTML = baukit.show.html(*self.logit_lens_layout)

        self.rewrite_button.style += baukit.show.Style(display='none')
        self.rewrite_info.innerHTML = 'Rewriting...'

        self.model = rewrite(layers, token.value, token_input.value, self.prompt, self.model, self.tokenizer, self.model_name)
        
        self.run_logit_lens(self, self.prompt, topk=5)

    def display_logit_lens(self):

        if self.hidden_state_function_dropdown.value == "Layer":
            color = [0, 0, 255]
        elif self.hidden_state_function_dropdown.value == "Layer Delta":
            color = [255, 0, 255]
        elif self.hidden_state_function_dropdown.value == "MLP":
            color = [0, 255, 0]
        elif self.hidden_state_function_dropdown.value == "Attn":
            color = [255, 0, 0]
            
        def color_fn(p):
            a = [int(255 * (1-p) + c * p) for c in color]
            return baukit.show.style(background=f'rgb({a[0]}, {a[1]}, {a[2]})')

        def hover(probabilities, words):
            lines = []
            for probability, word in zip(probabilities, words):
                lines.append(f'{word}: prob {probability:.2f}')
            return baukit.show.Attr(title='\n'.join(lines))
   
        # header line
        header_line = [ 
                [baukit.Div('Layer', style=self.bold_style)] + [baukit.Div(token, style=self.default_style + baukit.show.Style({'border-style' : 'solid'})) for token in self.prompt_tokens]
            ]

        logit_lens_layout = [header_line,
            # body
            [
                # first column
                [baukit.Div(str(layer), style=self.bold_style + baukit.show.Style({'border-style' : 'solid'}))] +
                [
                    baukit.Div(_words[0], style=self.default_style + color_fn(_probabilites[0]), attrs=hover(_probabilites, _words) + baukit.show.Attr(layer_idx=layer, token_idx=token_idx))
                    for token_idx, (_probabilites, _words) in enumerate(zip(probabilites, words))
                ]
            for layer, probabilites, words in zip(range(self.hidden_states.shape[0]), self.top_probs, self.top_words)],
            header_line
            ]

        self.logit_lens_layout = logit_lens_layout

        self.logit_lens.innerHTML = baukit.show.html(*logit_lens_layout)
        self.rewrite_button.style = baukit.show.style(display='block')


    def run_logit_lens(self, prompt, topk=5):

        if self.hidden_state_function_dropdown.value == "Layer":
            hidden_state_function = logitlens.get_hidden_state_layers
        elif self.hidden_state_function_dropdown.value == "Layer Delta":
            hidden_state_function = logitlens.get_hidden_state_layer_deltas
        elif self.hidden_state_function_dropdown.value == "MLP":
            hidden_state_function = logitlens.get_hidden_state_mlp
        elif self.hidden_state_function_dropdown.value == "Attn":
            hidden_state_function = logitlens.get_hidden_state_attn

        with torch.no_grad():
    
            self.prompt = prompt

            self.prompt_tokens = [self.tokenizer.decode(token) for token in self.tokenizer.encode(self.prompt)]

            hidden_states = hidden_state_function(self.model, self.tokenizer, prompt)

            self.predictions = self.decoder(hidden_states).cpu()

            self.hidden_states = hidden_states.cpu()

            self.prediction_probabilities = torch.nn.functional.softmax(self.predictions, dim=-1)

            self.top_probs, self.top_tokens = self.prediction_probabilities.topk(k=topk, dim=-1)
            self.top_probs, self.top_tokens = self.top_probs[:,0], self.top_tokens[:,0]

            self.top_words = [[[self.tokenizer.decode(token) for token in _tokens] for _tokens in tokens] for tokens in self.top_tokens]

            self.display_logit_lens()


def generate(model, tokenizer, text, num_generation=10):

    input = {k: torch.tensor(v)[None].cuda() for k, v in tokenizer(text).items()}

    for _ in range(num_generation):

        output = model(**input)
        logits = output['logits']

        prediction = logits[0, -1].argmax()

        input['input_ids'] = torch.cat((input['input_ids'], torch.tensor([prediction])[None].cuda()), dim=1)
        input['attention_mask'] = torch.cat((input['attention_mask'], torch.ones(1, 1).cuda()), dim=1)
    
    return tokenizer.decode(input['input_ids'][0, len(input['input_ids']):])

