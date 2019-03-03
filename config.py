# -*- coding: utf-8 -*-
class Config(object):
    train_file = "data/train.txt"
    dev_file = "data/valid.txt"
    test_file = "data/test.txt"
    embed_file = "data/glove.6B.100d.txt"


class LSTM_CRF_Config(Config):
    context_window = 1
    embed_dim = 100
    hidden_size = 150
    learning_rate = 0.0001
    use_char = False


config = {
    "lstm_crf": LSTM_CRF_Config,
}
