import sys
import os
import webbrowser
from circuitsvis.utils.render import render_local
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import Setup

from utils.model_config import load_model
from Interpretability import visualize_neuron_activation, display_attention_heads

model = load_model("pythia-70m")
text = "Which number is greater: 3 or 5?"

tokens = model.to_tokens(text)
logits, cache = model.run_with_cache(tokens)
layer = 0
position = 5;

#text_vis, topk_vis, neuron_acts, neuron_acts_rearranged = visualize_neuron_activation(cache, model, tokens)

vis = display_attention_heads(model, text, cache, layer=0, position=5)

html_str = render_local(vis)
save_path = f"attention_L{layer}_P{position}.html"
with open(save_path, "w") as f:
    f.write(html_str)

print(f"Visualization saved to: {os.path.abspath(save_path)}")

if True:
    webbrowser.open(f"file://{os.path.abspath(save_path)}", new=2)

'''
str_tokens = model.to_str_tokens(text)
for layer in range(model.cfg.n_layers):
    attention_pattern = cache["pattern", layer]
    display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern))
'''