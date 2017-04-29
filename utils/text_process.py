import jieba
from collections import Counter
import numpy as np

def encode_text(sentences, vocab=None, max_size=None):
    no_vocab = vocab is None
    tokens = []
    for s in sentences:
        words = [word for word in jieba.cut(s) if any('\u4e00' <= char <= '\u9fff' for char in word)]
        # words = [char for char in s if '\u4e00' <= char <= '\u9fff']
        tokens.append(['<BEG>'] + words + ['<END>'])
    if no_vocab:
        word_count = Counter(word for seq in tokens for word in seq)
        freq_list = sorted(word_count.items(), key=lambda p:p[1], reverse=True)
        if max_size is not None:
            top_words = [word for word, freq in freq_list[:max_size - 2]]
        else:
            top_words = [word for word, freq in freq_list]
        vocab = dict(zip(top_words, range(1, len(top_words) + 1)))
        vocab['<NUL>'] = 0
        vocab['<UNK>'] = len(vocab)
    ret = []
    for s in tokens:
        ret.append([vocab.get(word, vocab['<UNK>']) for word in s])
    vocab_inverse = dict((idx, word) for word, idx in vocab.items())
    if no_vocab:
        return ret, vocab, vocab_inverse
    else:
        return ret

def decode_text(seqs, vocab_inv):
    return [''.join(vocab_inv.get(idx) for idx in s) for s in seqs]

def seq2array(seqs, dtype=np.int64):
    max_len = max(len(s) for s in seqs)
    ret = np.zeros((len(seqs), max_len), dtype=dtype)
    seqlen = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        ret[i, :len(s)] = s
        seqlen[i] = len(s)
    return ret, seqlen

def array2seq(arr, stopper):
    ret = []
    for i in range(arr.shape[0]):
        pos_end = np.nonzero(arr[i, :] == stopper)[0]
        if pos_end.shape[0] > 0:
            ret.append(list(arr[i, :pos_end[0]]))
        else:
            ret.append(list(arr[i, :]))
    return ret
