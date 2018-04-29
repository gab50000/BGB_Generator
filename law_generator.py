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
NUM_LAYERS = 1  # lstm layers
HIDDEN_SIZE = 200
SAVE_FREQUENCY = 1000


class CharPredictor(torch.nn.Module):
    def __init__(self, number_of_chars, reverse_dict, num_layers):
        super().__init__()
        self.lstm = torch.nn.LSTM(number_of_chars, HIDDEN_SIZE, num_layers=num_layers)
        self.dense = torch.nn.Linear(HIDDEN_SIZE, number_of_chars)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.reverse_dict = reverse_dict
        self.dict = {v: k for k, v in self.reverse_dict.items()}
        self.number_of_chars = number_of_chars


    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        return self.softmax(self.dense(x)), hidden

    def sample(self, last_char, hidden, temperature):
        idx = self.dict[last_char]
        input = torch.zeros(1, 1, self.number_of_chars)
        input[0, 0, idx] = 1
        output, hidden = self.forward(input, hidden)
        probs = np.exp(output.detach().squeeze() / temperature).numpy()
        probs /= probs.sum()
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
    loss /= (len(sequence) - 1)
    return loss


def main():
    dataset = TextLoader("bgb.md")
    dataloader = DataLoader(dataset, batch_size=200)
    number_of_chars = dataset[0].shape[1]
    # lstm = torch.nn.LSTM(number_of_chars, number_of_chars)

    # print(lstm(dataset[:1]))  # optional: , (hidden, cell)))
    net = CharPredictor(number_of_chars, dataset.idx_to_char, NUM_LAYERS)
    if os.path.exists(MODEL_PATH):
        logger.info("Found model parameters. Will load them")
        net.load_state_dict(torch.load(MODEL_PATH))

    optimizer = torch.optim.Adam(params=net.parameters())
    criterion = torch.nn.NLLLoss()

    # output, (hidden, cell) = net(dataset[:1], (hidden, cell))

    while True:
        logger.info("Start from beginning of text")
        hidden, cell = (torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE),
                        torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE))
        for i, batch in enumerate(dataloader):
            # print("Batch:")
            # print(dataset.reverse(batch))
            loss = feed_sequence(net, hidden, cell, batch, criterion)
            print(f"Step {i: 8d}; Loss: {loss.item():12.8f}", end="\r")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % SAVE_FREQUENCY == 0:
                logger.info("Saving parameters")
                torch.save(net.state_dict(), MODEL_PATH)
                torch.save(net.state_dict(), MODEL_PATH + f"{i:05d}")


def run(filename):
    dataset = TextLoader("bgb.md")
    number_of_chars = 62
    net = CharPredictor(number_of_chars, dataset.idx_to_char, NUM_LAYERS)
    net.load_state_dict(torch.load(filename))

    hidden, cell = (torch.randn(NUM_LAYERS, 1, HIDDEN_SIZE),
                    torch.randn(NUM_LAYERS, 1, HIDDEN_SIZE))

    output = "a"
    temperature = 1

    while True:
        output, (hidden, cell) = net.sample(output, (hidden, cell), temperature)
        print(output, end="", flush=True)


fire.Fire()
