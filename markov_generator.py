from collections import defaultdict, Counter
import logging
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def chunks(list_, n):
    for i in range(0, len(list_)):
        yield list_[i : i + n]


class Generator:
    def __init__(self, n):
        self.n = n
        self.caches = [defaultdict(Counter) for _ in range(n)]
        self.logger = logging.getLogger(type(self).__name__)

    def predict(self, word_sequence: Tuple[str]) -> str:
        for cache in reversed(self.caches):
            if word_sequence in cache:
                return self.sample(cache[word_sequence])
        raise ValueError("Cannot sample")

    def sample(self, word_probs: Dict[str, float]) -> str:
        candidates = list(word_probs.keys())
        probs = list(word_probs.values())
        idx = np.argmax(np.random.multinomial(1, probs))
        return candidates[idx]

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
                    counter[key] = val / total


if __name__ == "__main__":
    with open("bgb.md", "r") as f:
        logging.basicConfig(level=logging.INFO)
        text = f.read().lower()
        gen = Generator(8)
        gen.train(list(text))

        seq = ("v", "e", "r")
        print("".join(seq), end="")
        while True:
            new_word = gen.predict(seq)
            seq = (*seq[1:], new_word)
            print(new_word, end="")
