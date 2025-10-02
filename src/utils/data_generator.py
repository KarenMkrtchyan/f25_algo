import random
from typing import Dict

class DataGenerator:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def _sample(self, low: int, high: int) -> Dict[str, object]:
        a = self.rng.randint(low, high)
        b = self.rng.randint(low, high)
        op = self.rng.choice(['>', '<'])
        label = (a > b) if op == '>' else (a < b)
        return {"text": f"{a} {op} {b}", "label": int(label), "a": a, "b": b, "op": op}

    def synthetic_data_one(self, size: int) -> list[dict]:
        return [self._sample(0, 10_000) for _ in range(size)]

    def synthetic_data_two(self, size: int) -> list[dict]:
        ranges = [(0, 9), (10, 99), (100, 999), (1000, 9999), (10000, 99999), (0, 99999)]
        chunks = size // len(ranges)
        data = []
        for low, high in ranges:
            for _ in range(chunks):
                data.append(self._sample(low, high))
        # pad if size not divisible
        while len(data) < size:
            data.append(self._sample(0, 99_999))
        return data
