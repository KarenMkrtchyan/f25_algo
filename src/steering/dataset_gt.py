# src/steering/dataset_gt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal
import numpy as np


def make_prompt(a: int, b: int) -> str:
    return f"Is {a} > {b}? Answer:"


@dataclass(frozen=True)
class Example:
    a: int
    b: int
    label: int          # int(a>b)
    clean: str          # "Is a > b?"
    corrupt: str        # "Is b > a?"


def build_dataset(
    n: int = 100,
    seed: int = 42,
    low: int = 1000,
    high: int = 9999,
) -> List[Example]:

    rng = np.random.default_rng(seed)
    data: List[Example] = []
    for _ in range(n):
        a = int(rng.integers(low, high + 1))
        b = int(rng.integers(low, a + 1))
        clean = make_prompt(a, b)
        corrupt = make_prompt(b, a)
        label = int(a > b)
        data.append(Example(a=a, b=b, label=label, clean=clean, corrupt=corrupt))
    return data


CategoryName = Literal["clean_true", "clean_false", "corrupt_true", "corrupt_false"]


def split_categories(examples: List[Example]) -> Dict[CategoryName, List[Example]]:
    """
    Splits into 4 categories:
      clean_true    : clean prompts where label==1 (a>b)
      clean_false   : clean prompts where label==0 (a<=b)
      corrupt_true  : corrupt prompts where label==1
      corrupt_false : corrupt prompts where label==0
    """
    out: Dict[CategoryName, List[Example]] = {
        "clean_true": [],
        "clean_false": [],
        "corrupt_true": [],
        "corrupt_false": [],
    }
    for ex in examples:
        if ex.label == 1:
            out["clean_true"].append(ex)
            out["corrupt_true"].append(ex)
        else:
            out["clean_false"].append(ex)
            out["corrupt_false"].append(ex)
    return out


def category_prompts(category: CategoryName, examples: List[Example]) -> List[str]:
    """Return the actual prompt strings for a category."""
    if category.startswith("clean"):
        return [ex.clean for ex in examples]
    return [ex.corrupt for ex in examples]
