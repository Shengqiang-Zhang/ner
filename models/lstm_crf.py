# -*- coding: utf-8 -*-

from datetime import timedelta, datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config
from .crf import CRF


class BiLSTM_CRF_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size,
                 out_size, learning_rate, dropout=0.3):
        super(BiLSTM_CRF_Model, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=True)
        self.out = nn.Linear(hidden_size * 2, out_size)
        self.crf = CRF(out_size)
        self.crf = self.crf.to(config.device)
        self.dropout = nn.Dropout(dropout)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def load_pre_trained(self, embed):
        self.embed = nn.Embedding.from_pretrained(embed, freeze=False)
        self.embed = self.embed.to(config.device)

    def forward(self, x, lens):
        B, T = x.shape
        x = self.embed(x)
        x = self.dropout(x)
        x = pack_padded_sequence(x, lens, batch_first=True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)

        return self.out(x)

    def train_iters(self, train_loader, dev_loader, test_loader,
                    epochs, interval, save_file):
        total_time = timedelta()
        max_acc, max_epoch = 0.0, 0
        for epoch in range(1, epochs + 1):
            start = datetime.now()
            print(f"Epoch: {epoch} / {epochs}:")
            self.train_single_iteration(train_loader)
            # Calculate train loss and accuracy
            train_loss, train_acc = self.evaluate(train_loader)
            print(f"{'train:':<6} Loss: {train_loss:.4f} Accuracy: {train_acc:.2%}")
            # Calculate dev loss and accuracy
            dev_loss, dev_acc = self.evaluate(dev_loader)
            print(f"{'dev:':<6} Loss: {dev_loss:.4f} Accuracy: {dev_acc:.2%}")
            # Calculate test loss and accuracy
            test_loss, test_acc = self.evaluate(test_loader)
            print(f"{'test:':<6} Loss: {test_loss:.4f} Accuracy: {test_acc:.2%}")

            time = datetime.now() - start
            print(f"{time}s elapsed.")
            total_time += time

            # Save the highest performance model
            if dev_acc > max_acc:
                torch.save(self, save_file)
                max_epoch, max_acc = epoch, dev_acc
            elif epoch - max_epoch >= interval:
                break

        print(f"Max accuracy of dev is {max_acc:.2%} at epoch {max_epoch}")
        print(f"Total time is {total_time}s")

    def train_single_iteration(self, train_loader):
        # Set the module in training mode
        self.train()

        for x, y, lens in train_loader:
            self.optimizer.zero_grad()

            # Set device option
            x = x.to(config.device)
            y = y.to(config.device)
            lens = lens.to(config.device)

            mask = x.gt(0)

            out = self.forward(x, lens)
            out = out.transpose(0, 1)  # [T, B, N]
            y, mask = y.t(), mask.t()  # [T, B]
            loss = self.crf(out, y, mask)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, loader):
        # Set the module in evaluation mode
        self.eval()
        # Compute loss and accuracy
        loss, tp, total = 0, 0, 0
        for x, y, lens in loader:
            # Set device options
            x = x.to(config.device)
            y = y.to(config.device)
            lens = lens.to(config.device)

            # Compute x > 0 element-wise
            mask = x.gt(0)

            target = y[mask]

            out = self.forward(x, lens)
            out = out.transpose(0, 1)  # [T, B, N]
            y, mask = y.t(), mask.t()  # [T, B]
            predict = self.crf.viterbi(out, mask)
            loss += self.crf(out, y, mask)
            # Compute precision
            tp += torch.sum(predict == target).item()
            total += lens.sum().item()
        avg_loss = loss / len(loader)
        return avg_loss, tp / total

    def collate_fn(self, data):
        x, y, lens = zip(
            *sorted(data, key=lambda x: x[-1], reverse=True)
        )
        max_len = lens[0]
        x = torch.stack(x)[:, :max_len]
        y = torch.stack(y)[:, :max_len]
        lens = torch.tensor(lens)

        return x, y, lens
