# FisherDataGen.py

import random
from typing import Dict, List, DefaultDict
from collections import defaultdict

class FisherDataGenerator:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def _build_number(self, tt: int, th: int, h: int, t: int, u: int) -> int:
        """Combine digits into a 5-digit number."""
        return tt * 10000 + th * 1000 + h * 100 + t * 10 + u

    def _sample_for_pair(
        self,
        pair_a: int,
        pair_b: int,
        position: str,
        n_per_class: int,
    ) -> List[Dict]:
        """Generate n_per_class examples with a fixed digit pair at a given position."""
        data = []
        for _ in range(n_per_class):
            # Random digits for A
            tt_a = self.rng.randint(0, 9)
            th_a = self.rng.randint(0, 9)
            h_a  = self.rng.randint(0, 9)
            t_a  = self.rng.randint(0, 9)
            u_a  = self.rng.randint(0, 9)

            # Random digits for B
            tt_b = self.rng.randint(0, 9)
            th_b = self.rng.randint(0, 9)
            h_b  = self.rng.randint(0, 9)
            t_b  = self.rng.randint(0, 9)
            u_b  = self.rng.randint(0, 9)

            # Override targeted position with (pair_a, pair_b)
            if position == "ten_thousands":
                tt_a, tt_b = pair_a, pair_b
            elif position == "thousands":
                th_a, th_b = pair_a, pair_b
            elif position == "hundreds":
                h_a,  h_b  = pair_a, pair_b
            elif position == "tens":
                t_a,  t_b  = pair_a, pair_b
            elif position == "units":
                u_a,  u_b  = pair_a, pair_b

            # Build full numbers
            a = self._build_number(tt_a, th_a, h_a, t_a, u_a)
            b = self._build_number(tt_b, th_b, h_b, t_b, u_b)

            # Comparison operator and label
            op = self.rng.choice(['>', '<'])
            label = (a > b) if op == '>' else (a < b)
            text = f"Is {a} > {b}? Answer: "

            data.append({
                "text": text,
                "label": label,
                "a": a,
                "b": b,
                "op": op,
            })

        return data

    def fisher_score_groups(
        self,
        n_per_class: int = 10,
    ) -> DefaultDict[str, DefaultDict[str, List[Dict]]]:
        """
        Generate n_per_class examples for each digit-pair in each position.
        Now supports 5-digit numbers.
        """
        positions = [
            "ten_thousands",
            "thousands",
            "hundreds",
            "tens",
            "units"
        ]

        groups: DefaultDict[str, DefaultDict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

        for pos in positions:
            for a_digit in range(10):
                for b_digit in range(10):
                    pair_class = f"{a_digit}{b_digit}"
                    examples = self._sample_for_pair(a_digit, b_digit, pos, n_per_class)
                    groups[pos][pair_class] = examples

        return groups
