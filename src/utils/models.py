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
        tokens = self.model.generate(**inputs)
        return self.tokenizer.decode(tokens[0])

if __name__ == "__main__":
    model = PythiaModel(
        model_name="EleutherAI/pythia-70m-deduped",
        revision="step3000",
        cache_dir="./pythia-70m-deduped/step3000"
    )
    
    output = model.generate("Hello, I am")
    print(output)
