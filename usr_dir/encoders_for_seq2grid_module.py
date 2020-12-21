from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensor2tensor.layers import common_layers
from tensor2tensor.models.transformer import (
    features_to_nonpadding,
    transformer_prepare_encoder,
    transformer_encoder,
)


def join_by_underbar(name1, name2):
    if len(name2) == 0:
        return name1
    else:
        return "_".join([name1, name2])

def gru_encode(input_seq, hparams, target_space, features, name, sequence_length=None, initial_state=None):
    if sequence_length==None:
        sequence_length = common_layers.length_from_embedding(input_seq)
    if initial_state!=None:
        initial_state = initial_state[:, 0, 0, :]

    input_seq = common_layers.flatten4d3d(input_seq)
    layers = [tf.nn.rnn_cell.GRUCell(hparams.hidden_size) for _ in range(hparams.num_hidden_layers)]
    with tf.variable_scope(name):
        # hidden_outputs (outputs of last layer) :  [batch_size, seq_len, hidden_size]
        # layer_final_output (layer-wise final outputs) : [num_hidden_layers, batch_size, hidden_size]
        hidden_outputs, layer_final_output = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.MultiRNNCell(layers),
            input_seq,
            sequence_length,
            initial_state=initial_state,
            dtype=tf.float32,
            time_major=False)
    hidden_outputs = tf.expand_dims(hidden_outputs, axis=-2)
    final_output = layer_final_output[-1]
    final_output = tf.expand_dims(final_output, axis=-2)
    final_output = tf.expand_dims(final_output, axis=-2)
    return hidden_outputs, final_output


def bid_gru_encode(input_seq, hparams, target_space, features, name, sequence_length=None):
    if sequence_length == None:
        sequence_length = common_layers.length_from_embedding(input_seq)
    input_seq = common_layers.flatten4d3d(input_seq)
    with tf.variable_scope(name):
        cell_fw = [tf.nn.rnn_cell.GRUCell(hparams.hidden_size) for _ in range(hparams.num_hidden_layers)]
        cell_bw = [tf.nn.rnn_cell.GRUCell(hparams.hidden_size) for _ in range(hparams.num_hidden_layers)]

        ((encoder_fw_outputs, encoder_bw_outputs),
         (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
            tf.nn.rnn_cell.MultiRNNCell(cell_fw),
            tf.nn.rnn_cell.MultiRNNCell(cell_bw),
            input_seq,
            sequence_length,
            initial_state_fw=None,
            initial_state_bw=None,
            dtype=tf.float32,
            time_major=False)

        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
        encoder_outputs = tf.expand_dims(encoder_outputs, axis=-2)

        final_output = tf.concat((encoder_fw_state[-1], encoder_bw_state[-1]),-1)
        final_output = tf.expand_dims(final_output, axis=-2)
        final_output = tf.expand_dims(final_output, axis=-2)
    return encoder_outputs, final_output

# return HIDDEN STATE in [-1, 1] if return_cell==False (default) else unbounded
def lstm_encode(input_seq, hparams, target_space, features, name, sequence_length=None):
    if sequence_length == None:
        sequence_length = common_layers.length_from_embedding(input_seq)
    input_seq = common_layers.flatten4d3d(input_seq)
    layers = [tf.nn.rnn_cell.LSTMCell(hparams.hidden_size) for _ in range(hparams.num_hidden_layers)]
    with tf.variable_scope(name):
        # hidden_outputs (outputs of last layer) :  [batch_size, seq_len, hidden_size]
        # layer_final_output (layer-wise final outputs) : [num_hidden_layers, 2, batch_size, hidden_size]
        hidden_outputs, layer_final_output = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.MultiRNNCell(layers),
            input_seq,
            sequence_length,
            initial_state=None,
            dtype=tf.float32,
            time_major=False)
    hidden_outputs = tf.expand_dims(hidden_outputs, axis=-2)
    c, h = layer_final_output[-1]
    final_output = h
    final_output = tf.expand_dims(final_output, axis=-2)
    final_output = tf.expand_dims(final_output, axis=-2)
    return hidden_outputs, final_output


def te_encode(input_seq, hparams, target_space, features, name):
    input_seq = common_layers.flatten4d3d(input_seq)

    (encoder_input, encoder_self_attention_bias, _) = (
        transformer_prepare_encoder(input_seq, target_space, hparams))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)
    encoder_output = transformer_encoder(
        encoder_input,
        encoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "input_seq"))
    encoder_output = tf.expand_dims(encoder_output, 2)
    return encoder_output

def mlp_encode(input_seq, hparams, name):
    encoded_input = input_seq
    for _ in range(hparams.num_hidden_layers):
        encoded_input = tf.layers.dense(encoded_input, hparams.hidden_size, activation=tf.nn.relu)
    return encoded_input


def encode_by_encoder_type(input_seq, hparams, target_space, features, seq_len=None):
    if hparams.encoder_type == "lstm":
        scope_name = "lstm_encoder"
        encoded_state_seq, _ = lstm_encode(input_seq, hparams, target_space, features, scope_name,
                                           sequence_length=seq_len)
    elif hparams.encoder_type == "gru":
        scope_name = "gru_encoder"
        encoded_state_seq, _ = gru_encode(input_seq, hparams, target_space, features, scope_name,
                                          sequence_length=seq_len)
    elif hparams.encoder_type == "gru_bid":
        scope_name = "gru_bid_encoder"
        encoded_state_seq, _ = bid_gru_encode(input_seq, hparams, target_space, features, scope_name,
                                              sequence_length=seq_len)
    elif hparams.encoder_type == "te":
        scope_name = "te_encoder"
        encoded_state_seq = te_encode(input_seq, hparams, target_space, features, scope_name)
    elif hparams.encoder_type == "mlp":
        scope_name = "mlp_encoder"
        encoded_state_seq = mlp_encode(input_seq, hparams, scope_name)
    else:  # if hparams.encoder_type == "id"
        encoded_state_seq = input_seq
    encoded_state_seq = tf.identity(encoded_state_seq, "encoded_state_seq")
    return encoded_state_seq


def encode_input_seq(input_seq, hparams, target_space, features, seq_len=None):
    encoded_state_seq = encode_by_encoder_type(input_seq, hparams, target_space, features, seq_len)
    return encoded_state_seq
