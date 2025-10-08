from transformers import GPTNeoXForCausalLM, AutoTokenizer

class PythiaModel:
    
    def __init__(self, model_name: str, revision: str, cache_dir: str):


        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir
        )
        self.model = GPTNeoXForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            pad_token_id=self.tokenizer.eos_token_id
        )

    def generate(self, prompt:str):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        tokens = self.model.generate(**inputs, max_new_tokens=64, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(tokens[0])


### 70m 
# model_naem "EleutherAI/pythia-70m-deduped",
# revision "step3000",
# cache_die "./pythia-70m-deduped/step3000"

