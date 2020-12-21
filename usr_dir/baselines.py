from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.utils import registry, t2t_model

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.keras.layers.core import *
from tensor2tensor.layers import common_layers, common_attention
from tensor2tensor.models import lstm, transformer, universal_transformer
from tensor2tensor.models.research import universal_transformer_util


class RMCell(rnn_cell_impl.LayerRNNCell):
    def __init__(self, num_units,
                 num_slots=1,
                 num_heads=16,
                 initializer=None,
                 forget_bias=1.0,
                 reuse=None, name=None, dtype=None, **kwargs):
        super(RMCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)
        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn("%s: Note that this cell is not optimized for performance. "
                         "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                         "performance on GPU.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._num_slots = num_slots
        self._num_heads = num_heads
        self._initializer = initializers.get(initializer)
        self._forget_bias = forget_bias
        self._activation = math_ops.tanh

        self._state_size = num_units
        self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        self._kernel = self.add_variable(
            rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 2 * self._num_units],
            initializer=self._initializer)
        if self.dtype is None:
            initializer = init_ops.zeros_initializer
        else:
            initializer = init_ops.zeros_initializer(dtype=self.dtype)
        self._bias = self.add_variable(
            rnn_cell_impl._BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=initializer)

        self.built = True

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid

        inputs_reshape = tf.expand_dims(inputs, axis=1)
        n = inputs_reshape.get_shape().as_list()[1]
        state_reshape = tf.reshape(state, [-1, self._num_slots, self._num_units])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        i_2d, f_2d = self.create_gates(inputs, state)

        m_prev_with_input = tf.concat([state_reshape, inputs_reshape], axis=1)
        state_tilde = self.attend_over_memory(m_prev_with_input)
        state_tilde = state_tilde[:, :-n, :]
        state = sigmoid(f_2d + self._forget_bias) * state_reshape + sigmoid(i_2d) * tf.tanh(state_tilde)

        state = tf.reshape(state, [-1, self._num_slots * self._num_units])
        # state = tf.ones_like(state)
        return state, state

    def create_gates(self, inputs, state):
        # i = input_gate, f = forget_gate
        concated = tf.concat([inputs, state], axis=-1)
        lstm_matrix = math_ops.matmul(concated, self._kernel)
        lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

        i, f = array_ops.split(value=lstm_matrix, num_or_size_splits=2, axis=1)
        return tf.expand_dims(i, axis=1), tf.expand_dims(f, axis=1)

    def multihead_attention(self, memory):
        seq_len = common_layers.shape_list(memory)[1]

        q = tf.layers.dense(memory, self._num_units, name="query")
        k = tf.layers.dense(memory, self._num_units, name="key")
        v = tf.layers.dense(memory, self._num_units, name="value")
        bias = None
        # bias = common_attention.attention_bias_lower_triangle(seq_len)
        q = common_attention.split_heads(q, self._num_heads)  # [batch_size, heads, q_len, hidden_size/heads]
        k = common_attention.split_heads(k, self._num_heads)
        v = common_attention.split_heads(v, self._num_heads)
        context = common_attention.dot_product_attention(q, k, v, bias)
        memory = common_attention.combine_heads(context)  # [batch_size, seq_len, hidden_size]
        return memory

    def attend_over_memory(self, memory):
        attended_memory = self.multihead_attention(memory)
        memory = common_layers.layer_norm(attended_memory + memory)

        # Add a skip connection to the attention_mlp's input.
        memory_mlp = tf.layers.dense(tf.layers.dense(memory, self._num_units), self._num_units)
        memory = common_layers.layer_norm(memory_mlp + memory)

        return memory

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "initializer": initializers.serialize(self._initializer),
            "forget_bias": self._forget_bias,
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(RMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _dropout_rm_cell(hparams, train):
    return tf.nn.rnn_cell.DropoutWrapper(
        RMCell(hparams.hidden_size),
        input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))


def rm(inputs, sequence_length, hparams, train, name, initial_state=None):
    layers = [_dropout_rm_cell(hparams, train)
              for _ in range(hparams.num_hidden_layers)]
    with tf.variable_scope(name):
        return tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.MultiRNNCell(layers),
            inputs,
            sequence_length,
            initial_state=initial_state,
            dtype=tf.float32,
            time_major=False)


def rm_seq2seq_internal(inputs, targets, hparams, train):
    with tf.variable_scope("rm_seq2seq"):
        if inputs is not None:
            inputs_length = common_layers.length_from_embedding(inputs)
            # Flatten inputs.
            inputs = common_layers.flatten4d3d(inputs)

            # LSTM encoder.
            inputs = tf.reverse_sequence(inputs, inputs_length, seq_axis=1)
            _, final_encoder_state = rm(inputs, inputs_length, hparams, train,
                                          "encoder")
        else:
            final_encoder_state = None

        # LSTM decoder.
        shifted_targets = common_layers.shift_right(targets)
        # Add 1 to account for the padding added to the left from shift_right
        targets_length = common_layers.length_from_embedding(shifted_targets) + 1
        decoder_outputs, _ = rm(
            common_layers.flatten4d3d(shifted_targets),
            targets_length,
            hparams,
            train,
            "decoder",
            initial_state=final_encoder_state)
        return tf.expand_dims(decoder_outputs, axis=2)


@registry.register_model
class RMCSeq2seq(t2t_model.T2TModel):

    def body(self, features):
        # TODO(lukaszkaiser): investigate this issue and repair.
        if self._hparams.initializer == "orthogonal":
            raise ValueError("LSTM models fail with orthogonal initializer.")
        train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
        return rm_seq2seq_internal(features.get("inputs"), features["targets"],
                                     self._hparams, train)


@registry.register_model
class LSTMToLabel(t2t_model.T2TModel):
    def body(self, features):
        inputs = features["inputs"]
        hparams= self._hparams
        train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
        with tf.variable_scope("lstm"):
            inputs_length = common_layers.length_from_embedding(inputs)
            inputs = common_layers.flatten4d3d(inputs)

            _, final_encoder_state = lstm.lstm(inputs, inputs_length, hparams, train, name="encoder")
            final_output = final_encoder_state[-1]
            c, h = final_output
            final_hidden_output = tf.expand_dims(h, axis=-2)
            final_hidden_output = tf.expand_dims(final_hidden_output, axis=-2)
            return final_hidden_output


@registry.register_model
class TransformerEncoderToLabel(t2t_model.T2TModel):
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderToLabel, self).__init__(*args, **kwargs)
        self._encoder_function = transformer.transformer_encoder
        self.attention_weights = {}  # For visualizing attention heads.

    def encode(self, inputs, target_space, hparams, features=None, losses=None):
        """Encode transformer inputs, see transformer_encode."""
        return transformer.transformer_encode(
            self._encoder_function, inputs, target_space, hparams,
            attention_weights=self.attention_weights,
            features=features, losses=losses)

    def body(self, features):
        hparams = self._hparams
        inputs = features["inputs"]
        target_space = features["target_space_id"]
        encoder_output, encoder_decoder_attention_bias = self.encode(
            inputs, target_space, hparams, features=features, losses=None)
        # encoder_output: [batch_size, input_length, hidden_dim]
        cls_output = encoder_output[:, :1, :]
        cls_output = tf.expand_dims(cls_output, axis=-2)
        return cls_output


class TransformerEncoder(transformer.TransformerEncoder):
    """Transformer, encoder only."""
    def body(self, features):
        hparams = self._hparams
        inputs = features["inputs"]
        target_space = features["target_space_id"]

        inputs = common_layers.flatten4d3d(inputs)

        (encoder_input, encoder_self_attention_bias, _) = (
            transformer.transformer_prepare_encoder(inputs, target_space, hparams))

        encoder_input = tf.nn.dropout(encoder_input,
                                      1.0 - hparams.layer_prepostprocess_dropout)
        encoder_output = transformer.transformer_encoder(
            encoder_input,
            encoder_self_attention_bias,
            hparams,
            nonpadding=transformer.features_to_nonpadding(features, "inputs"))

        encoder_output = encoder_output[:, :1, :]
        encoder_output = tf.expand_dims(encoder_output, 2)

        return encoder_output


@registry.register_model
class UniversalTransformerEncoderToLabel(universal_transformer.UniversalTransformerEncoder, t2t_model.T2TModel):
    def body(self, features):
        hparams = self._hparams

        assert self.has_input, ("universal_transformer_encoder is applicable on "
                                "problems with inputs")

        inputs = features["inputs"]
        target_space = features["target_space_id"]
        encoder_output, enc_extra_output = self.encode(
            inputs, target_space, hparams, features=features)

        encoder_output = encoder_output[:, :1, :]
        encoder_output = tf.expand_dims(encoder_output, 2)

        if hparams.recurrence_type == "act" and hparams.act_loss_weight != 0:
            ponder_times, remainders = enc_extra_output
            act_loss = hparams.act_loss_weight * tf.reduce_mean(ponder_times +
                                                                remainders)
            tf.contrib.summary.scalar("act_loss", act_loss)

            return encoder_output, {"act_loss": act_loss}
        return encoder_output

    def _greedy_infer(self, features, decode_length, use_tpu=False):
        return t2t_model.T2TModel._greedy_infer(self, features, decode_length)
