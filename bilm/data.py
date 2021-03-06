# originally based on https://github.com/tensorflow/models/tree/master/lm_1b
import glob
import random

import numpy as np

from typing import List



class Vocabulary(object):
    '''
    A token vocabulary.  Holds a map from token to ids and provides
    a method for encoding text to a sequence of ids.
    '''
    def __init__(self, filename, validate_file=False):
        '''
        filename = the vocabulary file.  It is a flat text file with one
            (normalized) token per line.  In addition, the file should also
            contain the special tokens <S>, </S>, <UNK> (case sensitive).
        '''
        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1

        with open(filename, encoding='utf-8') as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                if word_name == '<S>':
                    self._bos = idx
                elif word_name == '</S>':
                    self._eos = idx
                elif word_name == '<UNK>':
                    self._unk = idx
                if word_name == '!!!MAXTERMID':
                    continue

                self._id_to_word.append(word_name)
                self._word_to_id[word_name] = idx
                idx += 1
        
        # NOTE(feiga): add special token for permuted direction
        # <MD: MIDDLE> <SI: SIDE> is for "inward" "outward" directions
        # For example, a source sequence is "1,2,3,4,5,6"
        # "inward ": permuted sequence: "<SIDE> 1 6 2 5 3 4 <MIDDLE>"
        # "outward": permuted sequence: "<MIDDLE> 3 4 2 5 1 6 <SIDE>"
        word_name = "<MD>"
        self._mos = idx  # mos means "middle of sentence"
        self._id_to_word.append(word_name)
        self._word_to_id[word_name] = idx
        idx += 1

        word_name = "<SI>"
        self._sos = idx  # sos means "side of sentence"
        self._id_to_word.append(word_name)
        self._word_to_id[word_name] = idx
        idx += 1

        # <S2S: skip2start> <S2E: skip2end> is for "skip forward/backward" directions
        # Mainly to capture the phrase information (2-gram)
        # For example, a source sequencen is "1,2,3,4,5"
        # "Skip 2 forward ": permuted sequence: "<S2S> 1 3 5 2 4 <S2E>"
        # "Skip 2 backward": permuted sequence: "<S2E> 5 3 1 4 2 <S2S>"
        # For example, a source sequencen is "1,2,3,4,5,6"
        # "Skip 2 forward ": permuted sequence: "<S2S> 1 3 5 2 4 6 <S2E>"
        # "Skip 2 backward": permuted sequence: "<S2E> 6 4 2 5 3 1 <S2S>"
        word_name = "<S2S>"
        self._s2s = idx  # skip_bos means "Skip permuted begin of sentence"
        self._id_to_word.append(word_name)
        self._word_to_id[word_name] = idx
        idx += 1

        word_name = "<S2E>"
        self._s2e = idx  # skip_eos means "Skip permuted end of sentence"
        self._id_to_word.append(word_name)
        self._word_to_id[word_name] = idx
        idx += 1

        # <S3S: skip3start> <S3E: skip3end> is for "skip forward/backward" directions
        # Mainly to capture the phrase information (3-gram)
        # For example, a source sequencen is "1,2,3,4,5"
        # "Skip 3 forward ": permuted sequence: "<S3S> 1 4 2 5 3 <S3E>" ?
        # "Skip 3 backward": permuted sequence: "<S3E> 5 2 4 1 3 <S3S>" ?
        # For example, a source sequencen is "1,2,3,4,5,6"
        # "Skip 3 forward ": permuted sequence: "<S3S> 1 4 2 5 3 6 <S3E>"
        # "Skip 3 backward": permuted sequence: "<S3E> 6 3 5 2 4 1 <S3S>"
        word_name = "<S3S>"
        self._s3s = idx  # skip_bos means "Skip permuted begin of sentence"
        self._id_to_word.append(word_name)
        self._word_to_id[word_name] = idx
        idx += 1

        word_name = "<S3E>"
        self._s3e = idx  # skip_eos means "Skip permuted end of sentence"
        self._id_to_word.append(word_name)
        self._word_to_id[word_name] = idx
        idx += 1

        # check to ensure file has special tokens
        if validate_file:
            if self._bos == -1 or self._eos == -1 or self._unk == -1:
                raise ValueError("Ensure the vocabulary file has "
                                 "<S>, </S>, <UNK> tokens")

    @property
    def bos(self):
        return self._bos

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def sos(self):
        return self._sos

    @property
    def mos(self):
        return self._mos

    @property
    def s2s(self):
        return self._s2s

    @property
    def s2e(self):
        return self._s2e

    @property
    def s3s(self):
        return self._s3s

    @property
    def s3e(self):
        return self._s3e

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def decode(self, cur_ids):
        """Convert a list of ids to a sentence, with space inserted."""
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence, reverse=False, permuted=None, split=True):
        """Convert a sentence to a list of ids, with special tokens added.
        Sentence is a single string with tokens separated by whitespace.

        If reverse, then the sentence is assumed to be reversed, and
            this method will swap the BOS/EOS tokens appropriately."""

        if split:
            word_ids = [
                self.word_to_id(cur_word) for cur_word in sentence.split()
            ]
        else:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence]

        if reverse:
            return np.array([self.eos] + word_ids + [self.bos], dtype=np.int32)
        else:
            if permuted is not None:
                if permuted == 'inward':
                    return np.array([self.sos] + word_ids + [self.mos], dtype=np.int32)
                elif permuted == 'outward':
                    return np.array([self.mos] + word_ids + [self.sos], dtype=np.int32)
                elif permuted == 'skip2forward':
                    return np.array([self.s2s] + word_ids + [self.s2e], dtype=np.int32)
                elif permuted == 'skip2backward':
                    return np.array([self.s2e] + word_ids + [self.s2s], dtype=np.int32)
                elif permuted == 'skip3forward':
                    return np.array([self.s3s] + word_ids + [self.s3e], dtype=np.int32)
                elif permuted == 'skip3backward':
                    return np.array([self.s3e] + word_ids + [self.s3s], dtype=np.int32)
                else:
                    raise ValueError("Not implemented")
            else:
                return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)


class UnicodeCharsVocabulary(Vocabulary):
    """Vocabulary containing character-level and word level information.

    Has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.

    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.
    """
    def __init__(self, filename, max_word_length, **kwargs):
        super(UnicodeCharsVocabulary, self).__init__(filename, **kwargs)
        self._max_word_length = max_word_length

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.bos_char = 256  # <begin sentence>
        self.eos_char = 257  # <end sentence>
        self.bow_char = 258  # <begin word>
        self.eow_char = 259  # <end word>
        self.pad_char = 260  # <padding>
        self.mos_char = 261  # <middle of sentence>
        self.sos_char = 262  # <side of sentence>: similar to the start of the sentence
        self.s2s_char = 263  # <skip2start>
        self.s2e_char = 264  # <skip2end>
        self.s3s_char = 265  # <skip3start>
        self.s3e_char = 266  # <skip3end>

        num_words = len(self._id_to_word)

        self._word_char_ids = np.zeros([num_words, max_word_length],
            dtype=np.int32)

        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos(c):
            r = np.zeros([self.max_word_length], dtype=np.int32)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r
        self.bos_chars = _make_bos_eos(self.bos_char)
        self.eos_chars = _make_bos_eos(self.eos_char)
        self.mos_chars = _make_bos_eos(self.mos_char)
        self.sos_chars = _make_bos_eos(self.sos_char)
        self.s2s_chars = _make_bos_eos(self.s2s_char)
        self.s2e_chars = _make_bos_eos(self.s2e_char)
        self.s3s_chars = _make_bos_eos(self.s3s_char)
        self.s3e_chars = _make_bos_eos(self.s3e_char)

        for i, word in enumerate(self._id_to_word):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

        self._word_char_ids[self.bos] = self.bos_chars
        self._word_char_ids[self.eos] = self.eos_chars
        self._word_char_ids[self.mos] = self.mos_chars
        self._word_char_ids[self.sos] = self.sos_chars
        self._word_char_ids[self.s2s] = self.s2s_chars
        self._word_char_ids[self.s2e] = self.s2e_chars
        self._word_char_ids[self.s3s] = self.s3s_chars
        self._word_char_ids[self.s3e] = self.s3e_chars
        # TODO: properly handle <UNK>

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char

        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length-2)]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[k + 1] = self.eow_char

        return code

    def word_to_char_ids(self, word):
        if word in self._word_to_id:
            return self._word_char_ids[self._word_to_id[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence, reverse=False, permuted=False, split=True):
        '''
        Encode the sentence as a white space delimited string of tokens.
        '''
        if split:
            chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence]
        if reverse:
            return np.vstack([self.eos_chars] + chars_ids + [self.bos_chars])
        else:
            if permuted is not None:
                if permuted == 'inward':
                    return np.vstack([self.sos_chars] + chars_ids + [self.mos_chars])
                elif permuted == 'outward':
                    return np.vstack([self.mos_chars] + chars_ids + [self.sos_chars])
                elif permuted == "skip2forward":
                    return np.vstack([self.s2s_chars] + chars_ids + [self.s2e_chars])
                elif permuted == "skip2backward":
                    return np.vstack([self.s2e_chars] + chars_ids + [self.s2s_chars])
                elif permuted == "skip3forward":
                    return np.vstack([self.s3s_chars] + chars_ids + [self.s3e_chars])
                elif permuted == "skip3backward":
                    return np.vstack([self.s3e_chars] + chars_ids + [self.s3s_chars])
                else:
                    raise ValueError("Not implemented")
            else:
                return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])


class Batcher(object):
    ''' 
    Batch sentences of tokenized text into character id matrices.
    '''
    def __init__(self, lm_vocab_file: str, max_token_length: int):
        '''
        lm_vocab_file = the language model vocabulary file (one line per
            token)
        max_token_length = the maximum number of characters in each token
        '''
        self._lm_vocab = UnicodeCharsVocabulary(
            lm_vocab_file, max_token_length
        )
        self._max_token_length = max_token_length

    def batch_sentences(self, sentences: List[List[str]]):
        '''
        Batch the sentences as character ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        [['The', 'first', 'sentence', '.'], ['Second', '.']]
        '''
        n_sentences = len(sentences)
        max_length = max(len(sentence) for sentence in sentences) + 2

        X_char_ids = np.zeros(
            (n_sentences, max_length, self._max_token_length),
            dtype=np.int64
        )

        for k, sent in enumerate(sentences):
            length = len(sent) + 2
            char_ids_without_mask = self._lm_vocab.encode_chars(
                sent, split=False)
            # add one so that 0 is the mask value
            X_char_ids[k, :length, :] = char_ids_without_mask + 1

        return X_char_ids


class TokenBatcher(object):
    ''' 
    Batch sentences of tokenized text into token id matrices.
    '''
    def __init__(self, lm_vocab_file: str):
        '''
        lm_vocab_file = the language model vocabulary file (one line per
            token)
        '''
        self._lm_vocab = Vocabulary(lm_vocab_file)

    def batch_sentences(self, sentences: List[List[str]]):
        '''
        Batch the sentences as character ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        [['The', 'first', 'sentence', '.'], ['Second', '.']]
        '''
        n_sentences = len(sentences)
        max_length = max(len(sentence) for sentence in sentences) + 2

        X_ids = np.zeros((n_sentences, max_length), dtype=np.int64)

        for k, sent in enumerate(sentences):
            length = len(sent) + 2
            ids_without_mask = self._lm_vocab.encode(sent, split=False)
            # add one so that 0 is the mask value
            X_ids[k, :length] = ids_without_mask + 1

        return X_ids


##### for training
def _get_batch(generator, batch_size, num_steps, max_word_length):
    """Read batches of input."""
    cur_stream = [None] * batch_size

    no_more_data = False
    while True:
        inputs = np.zeros([batch_size, num_steps], np.int32)
        if max_word_length is not None:
            char_inputs = np.zeros([batch_size, num_steps, max_word_length],
                                np.int32)
        else:
            char_inputs = None
        targets = np.zeros([batch_size, num_steps], np.int32)

        for i in range(batch_size):
            cur_pos = 0

            while cur_pos < num_steps:
                if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
                    try:
                        cur_stream[i] = list(next(generator))
                    except StopIteration:
                        # No more data, exhaust current streams and quit
                        no_more_data = True
                        break

                how_many = min(len(cur_stream[i][0]) - 1, num_steps - cur_pos)
                next_pos = cur_pos + how_many

                inputs[i, cur_pos:next_pos] = cur_stream[i][0][:how_many]
                if max_word_length is not None:
                    char_inputs[i, cur_pos:next_pos] = cur_stream[i][1][
                                                                    :how_many]
                targets[i, cur_pos:next_pos] = cur_stream[i][0][1:how_many+1]

                cur_pos = next_pos

                cur_stream[i][0] = cur_stream[i][0][how_many:]
                if max_word_length is not None:
                    cur_stream[i][1] = cur_stream[i][1][how_many:]

        if no_more_data:
            # There is no more data.  Note: this will not return data
            # for the incomplete batch
            break

        X = {'token_ids': inputs, 'tokens_characters': char_inputs,
                 'next_token_id': targets}

        yield X


# TODO(permute list for a sequence)
def _permute_list(list, permute_pattern):
    permuted = []
    if permute_pattern == 'inward':
        i = 0
        j = len(list)-1
        for k in range(0, len(list)):
            if k % 2 == 0:
                permuted.append(list[i])
                i += 1
            else:
                permuted.append(list[j])
                j -= 1
    elif permute_pattern == 'outward':
        j = int(len(list)/2)
        i = int(len(list)/2) - 1
        for k in range(0, len(list)):
            if k % 2 == 0:
                permuted.append(list[j])
                j += 1
            else:
                permuted.append(list[i])
                i -= 1
    elif permute_pattern == "skip2forward":
        i = 0
        j = 1
        for k in range(i, len(list), 2):
            permuted.append(list[k])
        for k in range(j, len(list), 2):
            permuted.append(list[k])
    elif permute_pattern == "skip2backward":
        i = len(list) - 1
        j = len(list) - 2
        for k in range(i, -1, -2):
            permuted.append(list[k])
        for k in range(j, -1, -2):
            permuted.append(list[k])
    elif permute_pattern == "skip3forward":
        i = 0
        j = 1
        k = 2
        for m in range(i, len(list), 3):
            permuted.append(list[m])
        for m in range(j, len(list), 3):
            permuted.append(list[m])
        for m in range(k, len(list), 3):
            permuted.append(list[m])
    elif permute_pattern == "skip3backward":
        i = len(list) - 1
        j = len(list) - 2
        k = len(list) - 3
        for m in range(i, -1, -3):
            permuted.append(list[m])
        for m in range(j, -1, -3):
            permuted.append(list[m])
        for m in range(k, -1, -3):
            permuted.append(list[m])
    else:
        raise ValueError('Pattern error')
    return permuted


class LMDataset(object):
    """
    Hold a language model dataset.

    A dataset is a list of tokenized files.  Each file contains one sentence
        per line.  Each sentence is pre-tokenized and white space joined.
    """
    # NOTE(feiga): add param permuted, like the reverse
    def __init__(self, filepattern, vocab, reverse=False, permuted=None, test=False,
                 shuffle_on_load=False):
        '''
        filepattern = a glob string that specifies the list of files.
        vocab = an instance of Vocabulary or UnicodeCharsVocabulary
        reverse = if True, then iterate over tokens in each sentence in reverse
        test = if True, then iterate through all data once then stop.
            Otherwise, iterate forever.
        shuffle_on_load = if True, then shuffle the sentences after loading.
        '''
        self._vocab = vocab
        self._all_shards = glob.glob(filepattern)
        print('Found %d shards at %s' % (len(self._all_shards), filepattern))
        self._shards_to_choose = []

        self._reverse = reverse
        self._permuted = permuted
        self._test = test
        self._shuffle_on_load = shuffle_on_load
        self._use_char_inputs = hasattr(vocab, 'encode_chars')

        self._ids = self._load_random_shard()

    def _choose_random_shard(self):
        if len(self._shards_to_choose) == 0:
            self._shards_to_choose = list(self._all_shards)
            random.shuffle(self._shards_to_choose)
        shard_name = self._shards_to_choose.pop()
        return shard_name

    def _load_random_shard(self):
        """Randomly select a file and read it."""
        if self._test:
            if len(self._all_shards) == 0:
                # we've loaded all the data 
                # this will propogate up to the generator in get_batch
                # and stop iterating
                raise StopIteration
            else:
                shard_name = self._all_shards.pop()
        else:
            # just pick a random shard
            shard_name = self._choose_random_shard()

        ids = self._load_shard(shard_name)
        self._i = 0
        self._nids = len(ids)
        return ids

    def _load_shard(self, shard_name):
        """Read one file and convert to ids.

        Args:
            shard_name: file path.

        Returns:
            list of (id, char_id) tuples.
        """
        print('Loading data from: %s' % shard_name)
        with open(shard_name, encoding='utf-8') as f:
            sentences_raw = f.readlines()

        if self._reverse:
            sentences = []
            for sentence in sentences_raw:
                splitted = sentence.split()
                splitted.reverse()
                sentences.append(' '.join(splitted))
        # Note(feiga): process the permuted data
        elif self._permuted is not None:
            sentences = []
            # for sentence in ... :
            for sentence in sentences_raw:
                splitted = sentence.split()
                permuted = _permute_list(splitted, self._permuted)
                sentences.append(' '.join(permuted))
        else:
            sentences = sentences_raw
        
        if self._shuffle_on_load:
            random.shuffle(sentences)

        ids = [self.vocab.encode(sentence, self._reverse, self._permuted)
               for sentence in sentences]
        if self._use_char_inputs:
            chars_ids = [self.vocab.encode_chars(sentence, self._reverse, self._permuted)
                     for sentence in sentences]
        else:
            chars_ids = [None] * len(ids)

        print('Loaded %d sentences.' % len(ids))
        print('Finished loading')
        return list(zip(ids, chars_ids))

    def get_sentence(self):
        while True:
            if self._i == self._nids:
                self._ids = self._load_random_shard()
            ret = self._ids[self._i]
            self._i += 1
            yield ret

    @property
    def max_word_length(self):
        if self._use_char_inputs:
            return self._vocab.max_word_length
        else:
            return None

    def iter_batches(self, batch_size, num_steps):
        for X in _get_batch(self.get_sentence(), batch_size, num_steps,
                           self.max_word_length):

            # token_ids = (batch_size, num_steps)
            # char_inputs = (batch_size, num_steps, 50) of character ids
            # targets = word ID of next word (batch_size, num_steps)
            yield X

    @property
    def vocab(self):
        return self._vocab


class BidirectionalLMDataset(object):
    def __init__(self, filepattern, vocab, test=False, shuffle_on_load=False):
        '''
        bidirectional version of LMDataset
        '''
        self._data_forward = LMDataset(
            filepattern, vocab, reverse=False, test=test,
            shuffle_on_load=shuffle_on_load)
        self._data_reverse = LMDataset(
            filepattern, vocab, reverse=True, test=test,
            shuffle_on_load=shuffle_on_load)

    def iter_batches(self, batch_size, num_steps):
        max_word_length = self._data_forward.max_word_length

        for X, Xr in zip(
            _get_batch(self._data_forward.get_sentence(), batch_size,
                      num_steps, max_word_length),
            _get_batch(self._data_reverse.get_sentence(), batch_size,
                      num_steps, max_word_length)
            ):

            for k, v in Xr.items():
                X[k + '_reverse'] = v

            yield X


# NOTE(feiga): Dataset for more directions beyond bidirectionial lstm
class MultidirectionalLMDataset(object):
    def __init__(self, filepattern, vocab, permute_number, test=False, shuffle_on_load=False):
        '''
        multidirectional version of LMDataset
        '''
        # NOTE(lijun): add permute number
        self._permute_number = permute_number

        # NOTE(feiga): More dataset
        self._data_forward = LMDataset(
            filepattern, vocab, reverse=False, test=test,
            shuffle_on_load=shuffle_on_load)
        self._data_reverse = LMDataset(
            filepattern, vocab, reverse=True, test=test,
            shuffle_on_load=shuffle_on_load)
        # TODO(lijun):
        if permute_number == 4:
            # TODO(feiga):
            self._data_permuted1 = LMDataset(
                filepattern, vocab, reverse=False, permuted='inward', test=test,
                shuffle_on_load=shuffle_on_load)
            self._data_permuted2 = LMDataset(
                filepattern, vocab, reverse=False, permuted='outward', test=test,
                shuffle_on_load=shuffle_on_load)
        elif permute_number == 6:
            self._data_permuted1 = LMDataset(
                filepattern, vocab, reverse=False, permuted='inward', test=test,
                shuffle_on_load=shuffle_on_load)
            self._data_permuted2 = LMDataset(
                filepattern, vocab, reverse=False, permuted='outward', test=test,
                shuffle_on_load=shuffle_on_load)
            self._data_permuted3 = LMDataset(
                filepattern, vocab, reverse=False, permuted="skip2forward", test=test,
                shuffle_on_load=shuffle_on_load)
            self._data_permuted4 = LMDataset(
                filepattern, vocab, reverse=False, permuted="skip2backward", test=test,
                shuffle_on_load=shuffle_on_load)
        elif permute_number == 8:
            self._data_permuted1 = LMDataset(
                filepattern, vocab, reverse=False, permuted='inward', test=test,
                shuffle_on_load=shuffle_on_load)
            self._data_permuted2 = LMDataset(
                filepattern, vocab, reverse=False, permuted='outward', test=test,
                shuffle_on_load=shuffle_on_load)
            self._data_permuted3 = LMDataset(
                filepattern, vocab, reverse=False, permuted="skip2forward", test=test,
                shuffle_on_load=shuffle_on_load)
            self._data_permuted4 = LMDataset(
                filepattern, vocab, reverse=False, permuted="skip2backward", test=test,
                shuffle_on_load=shuffle_on_load)
            self._data_permuted5 = LMDataset(
                filepattern, vocab, reverse=False, permuted="skip3forward", test=test,
                shuffle_on_load=shuffle_on_load)
            self._data_permuted6 = LMDataset(
                filepattern, vocab, reverse=False, permuted="skip3backward", test=test,
                shuffle_on_load=shuffle_on_load)
        else:
            raise ValueError('Not implemented.')

    def iter_batches(self, batch_size, num_steps):
        max_word_length = self._data_forward.max_word_length

        # NOTE (lijun): different permute number to combine data

        if self._permute_number == 4:
            # NOTE(feiga): get batches from every datasets
            # TODO(feiga): for X, Xr, Xp1, Xp2...
            for X, Xr, Xp1, Xp2 in zip(
                _get_batch(self._data_forward.get_sentence(), batch_size,
                          num_steps, max_word_length),
                _get_batch(self._data_reverse.get_sentence(), batch_size,
                          num_steps, max_word_length),
                # Note(feiga):
                # add for permuted data
                _get_batch(self._data_permuted1.get_sentence(), batch_size,
                          num_steps, max_word_length),
                _get_batch(self._data_permuted2.get_sentence(), batch_size,
                         num_steps, max_word_length)
            ):
                # TODO(feiga): for X, Xr, Xp1, Xp2...
                for k, v in Xr.items():
                    X[k + '_reverse'] = v
                for k, v in Xp1.items():
                    X[k + '_permuted1'] = v
                for k, v in Xp2.items():
                    X[k + '_permuted2'] = v

                yield X

        elif self._permute_number == 6:
            for X, Xr, Xp1, Xp2, Xp3, Xp4 in zip(
                    _get_batch(self._data_forward.get_sentence(), batch_size,
                               num_steps, max_word_length),
                    _get_batch(self._data_reverse.get_sentence(), batch_size,
                               num_steps, max_word_length),
                    # Note(lijun): add permute3 and permute4
                    _get_batch(self._data_permuted1.get_sentence(), batch_size,
                               num_steps, max_word_length),
                    _get_batch(self._data_permuted2.get_sentence(), batch_size,
                               num_steps, max_word_length),
                    _get_batch(self._data_permuted3.get_sentence(), batch_size,
                               num_steps, max_word_length),
                    _get_batch(self._data_permuted4.get_sentence(), batch_size,
                               num_steps, max_word_length)
            ):
                for k, v in Xr.items():
                    X[k + '_reverse'] = v
                for k, v in Xp1.items():
                    X[k + '_permuted1'] = v
                for k, v in Xp2.items():
                    X[k + '_permuted2'] = v
                for k, v in Xp3.items():
                    X[k + '_permuted3'] = v
                for k, v in Xp4.items():
                    X[k + '_permuted4'] = v

                yield X

        elif self._permute_number == 8:
            for X, Xr, Xp1, Xp2, Xp3, Xp4, Xp5, Xp6 in zip(
                    _get_batch(self._data_forward.get_sentence(), batch_size,
                               num_steps, max_word_length),
                    _get_batch(self._data_reverse.get_sentence(), batch_size,
                               num_steps, max_word_length),
                    # Note(lijun): add permute3, 4, 5, 6
                    _get_batch(self._data_permuted1.get_sentence(), batch_size,
                               num_steps, max_word_length),
                    _get_batch(self._data_permuted2.get_sentence(), batch_size,
                               num_steps, max_word_length),
                    _get_batch(self._data_permuted3.get_sentence(), batch_size,
                               num_steps, max_word_length),
                    _get_batch(self._data_permuted4.get_sentence(), batch_size,
                               num_steps, max_word_length),
                    _get_batch(self._data_permuted5.get_sentence(), batch_size,
                               num_steps, max_word_length),
                    _get_batch(self._data_permuted6.get_sentence(), batch_size,
                               num_steps, max_word_length)
            ):
                for k, v in Xr.items():
                    X[k + '_reverse'] = v
                for k, v in Xp1.items():
                    X[k + '_permuted1'] = v
                for k, v in Xp2.items():
                    X[k + '_permuted2'] = v
                for k, v in Xp3.items():
                    X[k + '_permuted3'] = v
                for k, v in Xp4.items():
                    X[k + '_permuted4'] = v
                for k, v in Xp5.items():
                    X[k + '_permuted5'] = v
                for k, v in Xp6.items():
                    X[k + '_permuted6'] = v

                yield X
        else:
            raise ValueError('Not implemented.')

