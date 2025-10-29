import random
from typing import Dict, List, DefaultDict
from collections import defaultdict

class FisherDataGenerator:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def _build_number(self, hundreds: int, tens: int, units: int) -> int:
        """Combine digits into a 3-digit number."""
        return hundreds * 100 + tens * 10 + units

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
            # Randomly sample remaining digits
            h_a, t_a, u_a = self.rng.randint(0, 9), self.rng.randint(0, 9), self.rng.randint(0, 9)
            h_b, t_b, u_b = self.rng.randint(0, 9), self.rng.randint(0, 9), self.rng.randint(0, 9)

            # Replace the fixed position with our class pair
            if position == "hundreds":
                h_a, h_b = pair_a, pair_b
            elif position == "tens":
                t_a, t_b = pair_a, pair_b
            elif position == "units":
                u_a, u_b = pair_a, pair_b

            a = self._build_number(h_a, t_a, u_a)
            b = self._build_number(h_b, t_b, u_b)

            op = self.rng.choice(['>', '<'])
            label = (a > b) if op == '>' else (a < b)

            data.append({
                "text": f"{a} {op} {b}",
                "label": label,
                "a": a,
                "b": b,
                "op": op,
            })
        return data

    def fisher_score_groups(
        self,
        n_per_class: int = 10,
        low: int = 0,
        high: int = 999
    ) -> DefaultDict[str, DefaultDict[str, List[Dict]]]:
        """
        Generate exactly n_per_class examples for each digit-pair in each position.
        Returns:
        {
          'hundreds': {'00': [...], '01': [...], ...},
          'tens': {...},
          'units': {...}
        }
        """
        positions = ["hundreds", "tens", "units"]
        groups: DefaultDict[str, DefaultDict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

        for pos in positions:
            for a_digit in range(10):
                for b_digit in range(10):
                    pair_class = f"{a_digit}{b_digit}"
                    examples = self._sample_for_pair(a_digit, b_digit, pos, n_per_class)
                    groups[pos][pair_class] = examples

        return groups
