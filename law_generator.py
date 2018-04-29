import os
import pickle

import daiquiri
import fire
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


logger = daiquiri.getLogger(__name__)
daiquiri.setup(level=daiquiri.logging.DEBUG)


MODEL_PATH = "weights"


class CharPredictor(torch.nn.Module):
    def __init__(self, number_of_chars, reverse_dict):
        super().__init__()
        self.lstm = torch.nn.LSTM(number_of_chars, number_of_chars)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.reverse_dict = reverse_dict
        self.dict = {v: k for k, v in self.reverse_dict.items()}
        self.number_of_chars = number_of_chars


    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        return self.softmax(x), hidden

    def sample(self, last_char, hidden):
        idx = self.dict[last_char]
        input = torch.zeros(1, 1, self.number_of_chars)
        input[0, 0, idx] = 1
        output, hidden = self.forward(input, hidden)
        probs = np.exp(output.detach().squeeze()).numpy()
        idx = np.random.choice(range(self.number_of_chars), p=probs)
        return self.reverse_dict[max(idx, 1)], hidden



class TextLoader(Dataset):
    def __init__(self, filename):
        torch_fname = filename.rstrip(".torch") + ".torch"
        if os.path.exists(torch_fname):
            self.data = torch.load(torch_fname)
            with open("char_idx_dicts", "rb") as f:
                self.char_to_idx, self.idx_to_char = pickle.load(f)
        else:
            text = self.get_text(filename)
            self.data = self.text_to_onehot(text)
            torch.save(self.data, torch_fname)
            with open("char_idx_dicts", "wb") as f:
                pickle.dump((self.char_to_idx, self.idx_to_char), f)

    @staticmethod
    def get_text(filename):
        logger.info("Reading text")
        with open(filename, "r") as f:
            text = f.read().lower()
        return text

    def text_to_onehot(self, text):
        logger.info("Converting text to vector")
        text = text.lower()
        chars = list(set(text))
        text_len = len(text)
        logger.debug("Chars: %s", chars)
        char_to_idx = {char: i for i, char in enumerate(chars)}
        idx_to_char = {i: char for i, char in enumerate(chars)}
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char

        text_vec = torch.zeros(len(text), 1, len(chars), dtype=torch.float32)
        for i in tqdm(range(len(text))):
            char = text[i]
            text_vec[i][0][char_to_idx[char]] = 1

        return text_vec

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]


def feed_sequence(net, hidden, cell, sequence, criterion):
    loss = 0
    for in_, expected in zip(sequence[:-1], sequence[1:]):
        output, (hidden, cell) = net(in_[None, ...], (hidden, cell))
        loss += criterion(output[0], torch.argmax(expected, dim=-1))
    return loss


def main():
    dataset = TextLoader("bgb.md")
    dataloader = DataLoader(dataset, batch_size=200)
    number_of_chars = dataset[0].shape[1]
    # lstm = torch.nn.LSTM(number_of_chars, number_of_chars)

    # print(lstm(dataset[:1]))  # optional: , (hidden, cell)))
    net = CharPredictor(number_of_chars, dataset.idx_to_char)
    if os.path.exists(MODEL_PATH):
        logger.info("Found model parameters. Will load them")
        net.load_state_dict(torch.load(MODEL_PATH))

    optimizer = torch.optim.SGD(params=net.parameters(), lr=1e-4)
    criterion = torch.nn.NLLLoss()

    # output, (hidden, cell) = net(dataset[:1], (hidden, cell))

    while True:
        hidden, cell = (torch.zeros(1, 1, number_of_chars),
                        torch.zeros(1, 1, number_of_chars))
        for i, batch in enumerate(dataloader):
            # print("Batch:")
            # print(dataset.reverse(batch))
            loss = feed_sequence(net, hidden, cell, batch, criterion)
            print(f"Step {i: 8d}; Loss: {loss.item():10.2f}", end="\r")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                logger.info("Saving parameters")
                torch.save(net.state_dict(), MODEL_PATH + f"{i:05d}")
            elif i % 100 == 0:
                logger.info("Saving parameters")
                torch.save(net.state_dict(), MODEL_PATH)


def run(filename):
    dataset = TextDataset("bgb.md")
    number_of_chars = 63
    net = CharPredictor(number_of_chars, dataset.reverse_dict)
    net.load_state_dict(torch.load(filename))

    hidden, cell = (torch.randn(1, 1, number_of_chars),
                    torch.randn(1, 1, number_of_chars))

    output = "a"

    while True:
        output, (hidden, cell) = net.sample(output, (hidden, cell))
        print(output, end="")


fire.Fire()
