from transformer_lens import HookedTransformer
from tl_eval.lm_evaluator import evaluate_lm_eval

if __name__ == "__main__":
    # Load base model
    model = HookedTransformer.from_pretrained("pythia-70m")
    res = evaluate_lm_eval(model, tasks=["arc_easy"])
    print(res["results"])
