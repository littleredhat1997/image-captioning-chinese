import tensorflow as tf
import tensorlayer as tl
import numpy as np
from datetime import datetime
import os

from utils.load_data import *
from utils.text_process import *
from utils.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('n_epoch', 10, 'Number of epoches to train')
flags.DEFINE_integer('n_hidden', 512, 'Dimension of hidden states in LSTM')
flags.DEFINE_integer('n_embed', 512, 'Dimension of word embedding vectors')
flags.DEFINE_integer('k_beam', 3, 'Width of beam in beam search')
flags.DEFINE_float('end_penalty', 1.0, 'Penalty of each timestep after <END> in beam search')
flags.DEFINE_boolean('use_partial_val', False, 'Set this only when comparing results with validation set agreement')
flags.DEFINE_boolean('print_val', True, 'Print generated captions for selected images of validation set in log')

d_feature = 4096
n_epoch = FLAGS.n_epoch
n_hidden = FLAGS.n_hidden
n_embed = FLAGS.n_embed
keep_prob = 0.5
k_beam = FLAGS.k_beam
end_penalty = FLAGS.end_penalty
use_partial_val = FLAGS.use_partial_val
print_val = FLAGS.print_val

train_idx, train_sentences = load_text('./data/train.txt')
train_dict = {}
for idx, stc in zip(train_idx, train_sentences):
    train_dict.setdefault(idx, []).append(stc)
for idx in train_dict.keys():
    if len(train_dict[idx]) < 5:
        orig = train_dict[idx][:]
        shuffle_idx = np.random.permutation(len(orig))
        for i in range(5 - len(orig)):
            train_dict[idx].append(orig[shuffle_idx[i % len(orig)]])
train_pairs = [(idx, stc) for idx, stcs in train_dict.items() for stc in stcs]
train_idx = np.array([idx for idx, stc in train_pairs])
train_sentences = [stc for idx, stc in train_pairs]
train_seq, vocab, vocab_inv = encode_text(train_sentences)

val_idx, val_sentences = load_text('./data/valid.txt')
val_seq = encode_text(val_sentences, vocab=vocab, ignore_non_chinese=False, with_begin_end=False)
val_dict = {}
val_test = {}
for idx, seq in zip(val_idx, val_seq):
    if val_dict.get(idx) is None:
        if use_partial_val:
            val_test[idx] = seq
            val_dict[idx] = []
        else:
            val_dict[idx] = [seq]
    else:
        val_dict[idx].append(seq)
for idx in val_dict.keys():
    if len(val_dict[idx]) == 0:
        val_dict[idx] = [val_test[idx]]

idf = build_idf(val_dict)

train_seq, train_len = seq2array(train_seq)
max_step = train_seq.shape[1]
n_train = len(train_idx)

sess = tf.InteractiveSession()

img = tf.placeholder(tf.float32, shape=[None, d_feature], name='img')
seq_in = tf.placeholder(tf.int64, shape=[None, None], name='seq_in')
seq_len = tf.placeholder(tf.int64, shape=[None], name='seq_len')
seq_max_len = tf.shape(seq_in)[1]
seq_truth = tf.placeholder(tf.int64, shape=[None, None], name='seq_truth')

init_net = tl.layers.InputLayer(inputs=img)
init_net = tl.layers.DenseLayer(init_net,
    n_units=n_hidden,
    act=tf.sigmoid,
    name='init_transform')

network = tl.layers.EmbeddingInputlayer(
    inputs=seq_in,
    vocabulary_size=len(vocab),
    embedding_size=n_embed,
    name='embedding')
network = tl.layers.DropoutLayer(network, keep=keep_prob, name='lstm_in_dropout')
init_state_c = tf.placeholder(tf.float32, shape=[None, n_hidden], name='init_state_c')
init_state_h = tf.placeholder_with_default(init_net.outputs, shape=[None, n_hidden], name='init_state_h')
network = tl.layers.DynamicRNNLayer(network,
    cell_fn=tf.contrib.rnn.BasicLSTMCell,
    n_hidden=n_hidden,
    sequence_length=seq_len,
    initial_state=tf.contrib.rnn.LSTMStateTuple(init_state_c, init_state_h),
    name='lstm')
state_outputs = network.final_state
network = tl.layers.DropoutLayer(network, keep=keep_prob, name='lstm_out_dropout')
network = tl.layers.ReshapeLayer(network, shape=[-1, n_hidden], name='lstm_out_reshape')
network = tl.layers.DenseLayer(network,
    n_units=len(vocab),
    act=tf.identity,
    name='unembedding')
network = tl.layers.ReshapeLayer(network, shape=[-1, seq_max_len, len(vocab)], name='unembed_reshape')

k_top = tf.placeholder_with_default(1, shape=[], name='k_top')
top_k_loglike, top_k_ind = tf.nn.top_k(tf.nn.log_softmax(network.outputs), k=k_top)
loss_per_word = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=seq_truth, logits=network.outputs, name='cross_entropy')
seq_mask = tf.sequence_mask(seq_len, max_step - 1, dtype=tf.float32)
loss = tf.reduce_sum(loss_per_word * seq_mask)

dropout_dict = {**network.all_drop, **init_net.all_drop}
train_vars = network.all_params + init_net.all_params
train_op = tf.train.AdamOptimizer().minimize(loss, var_list=train_vars)


def generate_seq(batch_img, k_beam=1):
    batch_size = batch_img.shape[0]
    first_dim = batch_size * k_beam
    batch_img_in = np.repeat(batch_img, k_beam, axis=0)
    batch_seq_in = np.zeros((first_dim, 1), dtype=np.int64)
    batch_seq_len = np.ones((first_dim,), dtype=np.int64)
    batch_res = np.zeros((first_dim, max_step - 1), dtype=np.int64)
    batch_loglike_sum = np.zeros((first_dim,), dtype=np.float32)
    zero_state = np.zeros((first_dim, n_hidden), dtype=np.float32)
    for t in range(max_step - 1):
        if t == 0:
            batch_seq_in[:, 0] = vocab['<BEG>']
            feed_dict = {img: batch_img_in, k_top: k_beam,
                         seq_in: batch_seq_in, seq_len: batch_seq_len, init_state_c: zero_state}
        else:
            batch_seq_in[:, 0] = new_step
            feed_dict = {img: batch_img_in, k_top: k_beam,
                         seq_in: batch_seq_in, seq_len: batch_seq_len, init_state_c: new_c, init_state_h: new_h}
        feed_dict.update(tl.utils.dict_to_one(dropout_dict))
        step_loglike, step_ind, step_state = sess.run([top_k_loglike, top_k_ind, state_outputs], feed_dict=feed_dict)

        new_batch_res = np.zeros((first_dim, max_step - 1), dtype=np.int64)
        new_batch_loglike_sum = np.zeros((first_dim,), dtype=np.float32)
        new_step = np.zeros((first_dim,), dtype=np.int64)
        new_c = np.zeros((first_dim, n_hidden), dtype=np.float32)
        new_h = np.zeros((first_dim, n_hidden), dtype=np.float32)
        for i in range(batch_size):
            if t > 0:
                for j in range(k_beam):
                    if batch_res[i * k_beam + j, t - 1] == vocab['<END>']:
                        step_loglike[i * k_beam + j, 0, 0] = -end_penalty
                        step_loglike[i * k_beam + j, 0, 1:] = -np.inf
                        step_ind[i * k_beam + j, 0, 0] = vocab['<END>']
            cand_loglike = batch_loglike_sum[i * k_beam:(i + 1) * k_beam].reshape((-1, 1)) \
                           + step_loglike[i * k_beam:(i + 1) * k_beam, 0, :]
            if t == 0:
                cand_top_k = np.arange(k_beam)
            else:
                cand_top_k = np.argsort(cand_loglike, axis=None)[-1:-k_beam-1:-1]
            for rank, ind in enumerate(cand_top_k):
                row = ind // k_beam
                col = ind % k_beam
                new_batch_res[i * k_beam + rank, :t] = batch_res[i * k_beam + row, :t]
                new_step[i * k_beam + rank] = step_ind[i * k_beam + row, 0, col]
                new_batch_loglike_sum[i * k_beam + rank] = cand_loglike[row, col]
                new_c[i * k_beam + rank] = step_state.c[i * k_beam + row, :]
                new_h[i * k_beam + rank] = step_state.h[i * k_beam + row, :]
        new_batch_res[:, t] = new_step
        batch_res = new_batch_res
        batch_loglike_sum = new_batch_loglike_sum
        all_end = np.all(np.any(batch_res == vocab['<END>'], axis=1))
        if all_end:
            # print(decode_text(array2seq(batch_res[2*k_beam:3*k_beam, :], vocab['<END>']), vocab_inv))
            # print(batch_loglike_sum[2*k_beam:3*k_beam])
            break
    best_res = batch_res[::k_beam, :]
    seqs = array2seq(best_res, vocab['<END>'])
    texts = decode_text(seqs, vocab_inv)
    return seqs, texts


saver = tf.train.Saver()
saver.restore(sess,tf.train.latest_checkpoint('./checkpoints'))


from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

def load(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    batch_img = model.predict(x).reshape((1, 4096))
    _, texts = generate_seq(batch_img, k_beam=k_beam)
    
    return ''.join(texts)
