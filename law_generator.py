import os
import pickle
import time

import daiquiri
import fire
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


logger = daiquiri.getLogger(__name__)
daiquiri.setup(level=daiquiri.logging.DEBUG)


MODEL_PATH = "weights"
NUM_LAYERS = 2  # lstm layers
HIDDEN_SIZE = 100
SAVE_FREQUENCY = 500


class CharPredictor(torch.nn.Module):
    def __init__(self, number_of_chars, reverse_dict, num_layers, dropout=0):
        super().__init__()
        self.lstm = torch.nn.LSTM(number_of_chars, HIDDEN_SIZE, num_layers=num_layers,
                                  dropout=dropout)
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
        return self.reverse_dict[idx], hidden


class TextLoader(Dataset):
    def __init__(self, filename):
        self.data = torch.load(filename)
        with open("char_idx_dicts", "rb") as f:
            self.char_to_idx, self.idx_to_char = pickle.load(f)

    @staticmethod
    def get_text(filename):
        logger.info("Reading text")
        with open(filename, "r") as f:
            text = f.read().lower()
        return text

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]


def text_to_onehot(*filenames, save_to="data.torch"):
    logger.info("Converting text to vector")
    texts = []

    for fn in filenames:
        with open(fn, "r") as f:
            texts.append(f.read())

    texts = [text.lower() for text in texts]

    chars = set()
    text_len = 0
    for text in texts:
        chars.update(text)
        text_len += len(text)

    # add marker for end of file
    chars.add("EOF")
    chars = sorted(chars)
    text_len += len(texts)

    logger.debug("Chars: %s", chars)
    char_to_idx = {char: i for i, char in enumerate(chars)}
    idx_to_char = {i: char for i, char in enumerate(chars)}

    with open("char_idx_dicts", "wb") as f:
        pickle.dump((char_to_idx, idx_to_char), f)

    text_vec = torch.zeros(text_len, 1, len(chars), dtype=torch.float32)
    vec_idx = 0
    for fn, text in zip(filenames, texts):
        logger.info(fn)
        for char in text:
            text_vec[vec_idx][0][char_to_idx[char]] = 1
            vec_idx += 1
        text_vec[vec_idx][0][char_to_idx["EOF"]] = 1
        vec_idx += 1

    torch.save(text_vec, save_to)

def feed_sequence(net, hidden, cell, sequence, criterion):
    loss = 0
    for in_, expected in zip(sequence[:-1], sequence[1:]):
        output, (hidden, cell) = net(in_[None, ...], (hidden, cell))
        loss += criterion(output[0], torch.argmax(expected, dim=-1))
    loss /= (len(sequence) - 1)
    return loss


def main():
    torch.random.manual_seed(0)
    dataset = TextLoader("bgb.md")
    dataloader = DataLoader(dataset, batch_size=200)
    number_of_chars = dataset[0].shape[1]
    # lstm = torch.nn.LSTM(number_of_chars, number_of_chars)

    # print(lstm(dataset[:1]))  # optional: , (hidden, cell)))
    net = CharPredictor(number_of_chars, dataset.idx_to_char, NUM_LAYERS, dropout=0.5)
    if os.path.exists(MODEL_PATH):
        logger.info("Found model parameters. Will load them")
        net.load_state_dict(torch.load(MODEL_PATH))

    optimizer = torch.optim.Adam(params=net.parameters())
    criterion = torch.nn.NLLLoss()

    # output, (hidden, cell) = net(dataset[:1], (hidden, cell))
    i = 0
    while True:
        logger.info("Start from beginning of text")
        hidden, cell = (torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE),
                        torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE))
        for _, batch in enumerate(dataloader):
            # print("Batch:")
            # print(dataset.reverse(batch))
            loss = feed_sequence(net, hidden, cell, batch, criterion)
            print(f"Step {i: 8d}; Loss: {loss.item():12.8f}", end="\r")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % SAVE_FREQUENCY == 0:
                logger.info("Saving parameters")
                parameter_info = f"{NUM_LAYERS}-{HIDDEN_SIZE}"
                torch.save(net.state_dict(), MODEL_PATH + parameter_info)
                torch.save(net.state_dict(), MODEL_PATH + parameter_info + f"-{i:05d}")

            i += 1


def run(filename, temperature=1, start_char="#", seed=None):
    if seed:
        torch.random.manual_seed(seed)
    with open("char_idx_dicts", "rb") as f:
        char_to_idx, idx_to_char = pickle.load(f)

    number_of_chars = len(char_to_idx)
    net = CharPredictor(number_of_chars, idx_to_char, NUM_LAYERS)
    net.load_state_dict(torch.load(filename))

    hidden, cell = (torch.randn(NUM_LAYERS, 1, HIDDEN_SIZE),
                    torch.randn(NUM_LAYERS, 1, HIDDEN_SIZE))

    output = start_char

    while True:
        output, (hidden, cell) = net.sample(output, (hidden, cell), temperature)
        print(output, end="", flush=True)
        time.sleep(0.03)


fire.Fire()
