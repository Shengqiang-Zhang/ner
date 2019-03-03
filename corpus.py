# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset


class Corpus:
    PAD = "<PAD>"
    UNK = "<UNK>"
    SOS = "<SOS>"
    EOS = "<EOS>"

    def __init__(self, data_file, embedding_file=None):
        self.sentences = self.preprocess(data_file)
        self.words, self.tags, self.chars = self.parse(self.sentences)
        self.words = self.words + [self.PAD, self.UNK, self.SOS, self.EOS]
        self.chars = self.chars + [self.PAD, self.UNK]

        # Build dictionary of words, tags and chars
        self.word_dict = {w: i for i, w in enumerate(self.words)}
        self.tag_dict = {t: i for i, t in enumerate(self.tags)}
        self.char_dict = {c: i for i, c in enumerate(self.chars)}

        self.embedding = self.get_embedding(embedding_file) \
            if embedding_file is not None else None

    def get_embedding(self, embedding_file):
        with open(embedding_file, "r", encoding="utf-8") as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        words, embed = zip(*[
            (split[0], list(map(float, split[1:]))) for split in splits
        ])
        self.extend_word_list(words)
        # Initialize word embeddings, set words not in original embeddings
        # to a initial configuration.
        embed = torch.tensor(embed, dtype=torch.float)
        embed_indexes = [self.word_dict[w] for w in words]
        extended_embed = torch.FloatTensor(len(self.words), embed.size(1))
        self.init_embedding(extended_embed)
        extended_embed[embed_indexes] = embed

        return extended_embed

    def load_corpus(self, data_file, use_char=False, context_window=1, max_len=10):
        sentences = self.preprocess(data_file)
        x, y, char_x, lens = [], [], [], []
        for word_seq, tag_seq in sentences:
            word_index_seq = [self.word_dict.get(w, self.word_dict[self.UNK])
                              for w in word_seq]
            tag_index_seq = [self.tag_dict[t] for t in tag_seq]

            # Obtain each word's contexts
            if context_window > 1:
                x.append(self.get_context(word_index_seq, context_window))
            else:
                x.append(torch.tensor(word_index_seq, dtype=torch.long))
            y.append(torch.tensor(tag_index_seq, dtype=torch.long))
            # Pad sentences shorter than the max_len with 0
            char_x.append(torch.tensor([
                [self.char_dict.get(c, self.char_dict[self.UNK]) for c in w[:max_len]]
                + [0] * (max_len - len(w)) for w in word_seq
            ]))
            lens.append(len(tag_index_seq))

        x = pad_sequence(x, True)
        y = pad_sequence(y, True)
        char_x = pad_sequence(char_x, True)
        lens = torch.tensor(lens)

        if use_char:
            dataset = TensorDataset(x, y, char_x, lens)
        else:
            dataset = TensorDataset(x, y, lens)
        return dataset

    def get_context(self, word_index_seq, context_window):
        # todo
        half = context_window // 2
        seq_len = len(word_index_seq)
        word_index_seq = [self.word_dict[self.SOS]] * half + word_index_seq + \
                         [self.word_dict[self.EOS]] * half
        context = [word_index_seq[i: i + context_window] for i in range(seq_len)]
        context = torch.tensor(context, dtype=torch.long)
        return context

    @staticmethod
    def init_embedding(tensor):
        std = (1. / tensor.size(1)) ** 0.5
        nn.init.normal_(tensor, mean=0, std=std)

    def extend_word_list(self, words):
        unk_words = [w for w in words if w not in self.word_dict]
        unk_chars = [c for c in ''.join(unk_words) if c not in self.char_dict]

        self.words = sorted(set(self.words + unk_words) - {self.PAD})
        self.chars = sorted(set(self.chars + unk_chars) - {self.PAD})
        self.words += [self.PAD]
        self.chars += [self.PAD]
        # Update dictionary
        self.word_dict = {w: i for i, w in enumerate(self.words)}
        self.char_dict = {c: i for i, c in enumerate(self.chars)}

    @staticmethod
    def preprocess(data_file):
        start = 0
        sentences = []
        with open(data_file, "r", encoding="utf-8") as f:
            lines = [line for line in f]
        for i, line in enumerate(lines):
            if len(lines[i]) <= 1:
                splits = [l.split()[0:4:3] for l in lines[start:i]]
                word_seq, tag_seq = zip(*splits)
                start = i + 1
                while start < len(lines) and len(lines[start]) <= 1:
                    start += 1
                sentences.append((word_seq, tag_seq))

        return sentences

    @staticmethod
    def parse(sentences):
        word_seqs, tag_seqs = zip(*sentences)
        words = sorted(set(w for word_seq in word_seqs for w in word_seq))
        tags = sorted(set(t for tag_seq in tag_seqs for t in tag_seq))
        chars = sorted(set(''.join(words)))

        return words, tags, chars

    def __repr__(self):
        info = f"{self.__class__.__name__}(\n"
        info += f"{'':2}num of sentences: {len(self.sentences)}\n"
        info += f"{'':2}num of words: {len(self.words)}\n"
        info += f"{'':2}num of tags: {len(self.tags)}\n"
        info += f"{'':2}num of chars: {len(self.chars)}\n"
        info += ")\n"
        return info
