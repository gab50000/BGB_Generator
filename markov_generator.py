from collections import defaultdict, Counter
import logging
import pathlib
import pickle
import re
import time
from typing import Dict, List, Tuple

import fire
import numpy as np
from tqdm import tqdm


def chunks(list_, n):
    for i in range(0, len(list_)):
        yield list_[i : i + n]


class Generator:
    def __init__(self, n, temperature=1):
        self.n = n
        self.temperature = temperature
        self.caches = [defaultdict(Counter) for _ in range(n)]
        self.logger = logging.getLogger(type(self).__name__)

    @classmethod
    def from_file(cls, filename) -> "Generator":
        with open(filename, "rb") as f:
            return pickle.load(f)

    def to_file(self, filename):
        self.logger.info("Save to file %s", filename)
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def predict(self, word_sequence: Tuple[str, ...]) -> str:
        for cache in reversed(self.caches):
            if word_sequence in cache:
                return self.sample(cache[word_sequence])
        raise ValueError("Cannot sample")

    def sample(self, log_probs: Dict[str, float]) -> str:
        candidates = list(log_probs.keys())
        probs = np.asfarray(list(log_probs.values()))
        probs = np.exp(probs / self.temperature)
        probs /= probs.sum()
        return np.random.choice(candidates, p=probs)

    def learn(self, word_sequence: List[str], following_word: str):
        self.caches[len(word_sequence) - 1][tuple(word_sequence)][following_word] += 1

    def train_len_i(self, i: int, input_text: List[str]):
        self.logger.info("Train on length %s", i)
        for chunk in tqdm(chunks(input_text, i + 1)):
            *prev_text, following_word = chunk
            self.learn(prev_text, following_word)

    def train(self, input_text: List[str]):
        for i in range(1, self.n + 1):
            self.train_len_i(i, input_text)
        self._normalize_counters()

    def _normalize_counters(self):
        self.logger.info("Normalize counters")
        for cache in self.caches:
            for counter in tqdm(cache.values()):
                total = sum(counter.values())
                for key, val in counter.items():
                    counter[key] = np.log(val / total)


def main(input_text, temperature):
    logging.basicConfig(level=logging.INFO)
    with open(input_text, "r") as f:
        text = f.read().lower()
        text = re.split(r"\t| ", re.sub(r"\n", " \n ", text))

    pickle_filename = "gen.pickle"
    if pathlib.Path(pickle_filename).exists():
        gen = Generator.from_file(pickle_filename)
    else:
        gen = Generator(10)
        gen.train(list(text))
        gen.to_file(pickle_filename)

    gen.temperature = temperature
    seq = ("der",)
    print("".join(seq), end=" ")
    while True:
        new_word = gen.predict(seq)
        seq = (*seq[1:], new_word)
        print(new_word, end=" ", flush=True)
        time.sleep(0.1)


fire.Fire(main)