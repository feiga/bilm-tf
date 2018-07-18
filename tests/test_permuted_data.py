from bilm.data import *

vocab = Vocabulary('vocab.test')
dataset = MultidirectionalLMDataset('data.test', vocab)
for X in dataset.iter_batches(1, 9):
    print(X['token_ids'][:])
    print(X['token_ids_reverse'])
    print(X['token_ids_permuted1'])
    print(X['token_ids_permuted2'])
    break


