
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset, MultidirectionalLMDataset


def main(args):
    options, ckpt_file = load_options_latest_checkpoint(args.save_dir)

    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = 128  # batch size for each GPU
    n_gpus = args.n_gpus
    permute_number = args.permute_number

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 768648884

    options = {
     'bidirectional': True,
     'multidirectional': True,
     'permute_number': permute_number,

     'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [[1, 32],
       [2, 32],
       [3, 64],
       [4, 128],
       [5, 256],
       [6, 512],
       [7, 1024]],
      'max_characters_per_token': 50,
      'n_characters': 267,  # NOTE (lijun): add more character tokens
      'n_highway': 2},
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': args.dim,  # NOTE(feiga): halved dimensions comparing with ELMo (default=2048)
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': args.projection_dim,  # NOTE(feiga): halved dimensions comparing with ELMo (default=256)
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': 10,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
     'n_negative_samples_batch': 8192,
    }

    prefix = args.train_prefix
    data = MultidirectionalLMDataset(prefix, vocab, permute_number, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir, permute_number,
          restart_ckpt_file=ckpt_file)
    # if ckpt_file exists, reload to train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--n_gpus', type=int, default=4, help='Number of gpu cards.')
    parser.add_argument('--permute_number', type=int, default=4, help='Number of permutations.')
    parser.add_argument('--dim', type=int, default=2048, help='Input dimension.')
    parser.add_argument('--projection_dim', type=int, default=256, help='Hidden dimension.')

    args = parser.parse_args()
    main(args)

