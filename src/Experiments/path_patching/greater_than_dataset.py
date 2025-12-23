import torch as t

class GreaterThanDataset:
    def __init__(self, model, prompts, labels, device):
        self.toks = model.to_tokens(prompts, prepend_bos=True).to(device)

        self.word_idx = {"end": self.toks.size(1) - 1}

        answers = [("Yes", "No") if label == "Yes" else ("No", "Yes") for label in labels]
        answer_tokens = t.concat([
            model.to_tokens(pair, prepend_bos=False).T for pair in answers
        ])

        self.io_tokenIDs = answer_tokens[:, 0]
        self.s_tokenIDs = answer_tokens[:, 1]

        self.answer_tokens = answer_tokens

    def gen_flipped_prompts(self, model: str):
        """
        Approximate IOI's abc_dataset: same prompts, but swap which answer is 'correct'.
        mode string is ignored but kept for API compatibility.
        """
        flipped_labels = ["No" if lbl == "Yes" else "Yes" for lbl in labels]

        flipped_answers = [("Yes", "No") if label == "Yes" else ("No", "Yes")
                           for label in flipped_labels]
        flipped_answer_tokens = t.concat([
            model.to_tokens(pair, prepend_bos=False).T for pair in flipped_answers
        ])

        flipped_dataset = GreaterThanDataset.__new__(GreaterThanDataset)
        flipped_dataset.toks = self.toks.clone()
        flipped_dataset.word_idx = self.word_idx.copy()
        flipped_dataset.io_tokenIDs = flipped_answer_tokens[:, 0]
        flipped_dataset.s_tokenIDs = flipped_answer_tokens[:, 1]
        flipped_dataset.answer_tokens = flipped_answer_tokens

        return flipped_dataset
