from typing import Dict, List

class ICLPrompt:

    def __init__(self, examples: List[str] = None):

        self.examples = examples or [
            "100 > 50: true",
            "25 < 30: true", 
            "80 > 90: false",
            "15 < 10: false",
            "200 > 150: true",
            "45 < 50: true",
            "300 > 100: true",
            "75 > 80: false",
            "75 > 30: true",
            "20 < 30: true",
            "98 > 100: false",
        ]

    def build_prompt(self, sample: Dict, num_examples: int = 5) -> str:
        prompt = "Evaluate if each comparison is true or false: \n"

        for ex in self.examples[:num_examples]:
            prompt += ex + "\n"

        prompt += f"{sample['text']}: "

        return prompt

