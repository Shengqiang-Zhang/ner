# -*- coding: utf-8 -*-
import argparse
from datetime import datetime, timedelta

import torch
from torch.utils.data import DataLoader

import config
from corpus import Corpus
from models import BiLSTM_CRF_Model

if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser(
        description="Create several models for Named Entity Recognition."
    )
    parser.add_argument('--model', '-m', default='lstm_crf',
                        choices=['lstm_crf', 'char_lstm_crf'],
                        help="Choose the model for NER")
    parser.add_argument("--epochs", action="store", default=50, type=int,
                        help="Set the max num of epochs")
    parser.add_argument("--batch_size", action="store", default=64, type=int,
                        help="Set the size of batch")
    parser.add_argument("--dropout", action="store", default=0.2, type=float,
                        help="Set the dropout ratio")
    parser.add_argument("--interval", action="store", default=10, type=int,
                        help="Set the max interval to stop")
    parser.add_argument("--learning_rate", action="store", default=0.001, type=float,
                        help="Set the learning rate of training")
    parser.add_argument("--threads", "-t", action="store", default=4, type=int,
                        help="Set the max num of threads")
    parser.add_argument("--seed", "-s", action="store", default=1, type=int,
                        help="Set the seed for generating random numbers")
    parser.add_argument("--save_file", "-f", action="store", default="network.pt",
                        help="Set model saving file")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    # Get model configuration
    config = config.config[args.model]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Preprocess the data ...")
    corpus = Corpus(data_file=config.train_file, embedding_file=config.embed_file)
    print(corpus)

    print("Load the dataset ...")
    train_set = corpus.load_corpus(config.train_file, config.use_char,
                                   config.context_window)
    dev_set = corpus.load_corpus(config.dev_file, config.use_char,
                                 config.context_window)
    test_set = corpus.load_corpus(config.test_file, config.use_char,
                                  config.context_window)
    print(f"{'':2}size of train set: {len(train_set)}\n"
          f"{'':2}size of dev set: {len(dev_set)}\n"
          f"{'':2}size of test set: {len(test_set)}")

    start_time = datetime.now()
    torch.manual_seed(args.seed)

    print("Building model ...")
    if args.model == "lstm_crf":
        print(f"{'':2}vocab_size: {len(corpus.words)}\n"
              f"{'':2}embed_dim: {config.embed_dim}\n"
              f"{'':2}hidden_size: {config.hidden_size}\n"
              f"{'':2}out_size: {len(corpus.tags)}\n")
        model = BiLSTM_CRF_Model(vocab_size=len(corpus.words),
                                 embed_dim=config.embed_dim,
                                 hidden_size=config.hidden_size,
                                 out_size=len(corpus.tags),
                                 learning_rate=config.learning_rate)
        model.to(device)

    model.load_pre_trained(corpus.embedding)
    print(f"{model}")

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=model.collate_fn)
    dev_loader = DataLoader(dataset=dev_set,
                            batch_size=args.batch_size,
                            collate_fn=model.collate_fn)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args.batch_size,
                             collate_fn=model.collate_fn)

    print("Training ...")
    print(f"{'':2}epochs: {args.epochs}\n"
          f"{'':2}batch_size: {args.batch_size}\n"
          f"{'':2}interval: {args.interval}\n"
          f"{'':2}learning_rate: {args.learning_rate}\n")
    model.train_iters(train_loader=train_loader,
                      dev_loader=dev_loader,
                      test_loader=test_loader,
                      device=device,
                      epochs=args.epochs,
                      interval=args.interval,
                      save_file=args.save_file)

    model = torch.load(args.save_file)
    loss, accuracy = model.evaluate(test_loader, device)
    print(f"{'test':<6} Loss: {loss:.4f} Accuracy: {accuracy:.2%}")
    print(f"{datetime.now() - start_time}s elapsed.")
