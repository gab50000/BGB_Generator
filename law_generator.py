import fire
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader


class CharPredictor(torch.nn.Module):
    def __init__(self, number_of_chars):
        super().__init__()
        self.lstm = torch.nn.LSTM(number_of_chars, number_of_chars)
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        return self.softmax(x), hidden


def get_text(filename):
    with open(filename, "r") as f:
        text = f.read().lower()
    return text


def text_to_onehot(text):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(text)
    text_vec = tokenizer.texts_to_matrix(text).astype(np.float32)
    reverse_dict = {v: k for k, v in tokenizer.word_index.items()}

    def vec_to_text(vec):
        return "".join(reverse_dict[char] for char in np.argmax(vec, axis=1))

    return vec_to_text, text_vec


class TextDataset(Dataset):
    def __init__(self, filename):
        bgb_text = get_text(filename)
        reverse, text_vec = text_to_onehot(bgb_text)
        self.reverse = reverse
        self.data = torch.from_numpy(text_vec).reshape(-1, 1, text_vec.shape[-1])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]


def main():
    dataset = TextDataset("bgb.md")
    number_of_chars = dataset[0].shape[1]
    # lstm = torch.nn.LSTM(number_of_chars, number_of_chars)

    hidden, cell = (torch.zeros(1, 1, number_of_chars),
                    torch.zeros(1, 1, number_of_chars))

    # print(lstm(dataset[:1]))  # optional: , (hidden, cell)))
    cp = CharPredictor(number_of_chars)

    print(cp(dataset[:1], (hidden, cell)))


fire.Fire()