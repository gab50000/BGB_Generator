from collections import defaultdict, Counter
import logging
from typing import List

from tqdm import tqdm


def chunks(list_, n):
    for i in range(0, len(list_)):
        yield list_[i : i + n]


class Generator:
    def __init__(self, n):
        self.n = n
        self.cache = defaultdict(Counter)
        self.logger = logging.getLogger(type(self).__name__)

    def predict(self, word_sequence: List[str]) -> str:
        pass

    def learn(self, word_sequence: List[str], following_word: str):
        self.cache[tuple(word_sequence)][following_word] += 1

    def train(self, input_text: List[str]):
        for chunk in tqdm(chunks(input_text, self.n + 1)):
            *prev_text, following_word = chunk
            self.learn(prev_text, following_word)

        self._normalize_counters()

    def _normalize_counters(self):
        self.logger.info("Normalize counters")
        for counter in tqdm(self.cache.values()):
            total = sum(counter.values())
            for key, val in counter.items():
                counter[key] = val / total


if __name__ == "__main__":
    with open("bgb.md", "r") as f:
        logging.basicConfig(level=logging.INFO)
        text = f.read().lower()
        gen = Generator(5)
        gen.train(list(text))
