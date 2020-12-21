from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras.layers.core import *
from tensor2tensor.layers import common_layers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.eager import context

# action (= input[:, :3]): top_list_update==0, new_list_update==1, no_op==2
# state: [batch_size, num_lists, list_size, num_units]
# internal_state: [batch_size, num_lists, list_size, num_units]
class NestedListOperationCell(rnn_cell_impl.LayerRNNCell):
    def __init__(self,
                 num_units,
                 list_size=10,
                 num_lists=10,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(NestedListOperationCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)

        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn("%s: Note that this cell is not optimized for performance. "
                         "Please use tf.contrib.cudnn_rnn.CudnnGRU for better "
                         "performance on GPU.", self)
        # Inputs must be 2-dimensional.
        self.input_spec = input_spec.InputSpec(ndim=2)

        self._num_units = num_units
        self._list_size = list_size
        self._num_lists = num_lists
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = tf.sigmoid
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)

    @property
    def state_size(self):
        return self._list_size * self._num_lists * self._num_units

    @property
    def output_size(self):
        return self._list_size * self._num_lists * self._num_units

    @property
    def num_lists(self):
        return self._num_lists

    @property
    def list_size(self):
        return self._list_size

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % str(inputs_shape))
        self.built = True

    def call(self, inputs, state):

        p_list = inputs[:, 1:2]
        p_state = inputs[:, 0:1]
        p_no = inputs[:, 2:3]
        x = inputs[:, 3:]
        x_expanded = tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)

        # internal_state: [batch_size, num_lists, list_size, num_units]
        batch_size = tf.shape(state)[0]
        state = tf.reshape(state, [batch_size, self._num_lists, self._list_size, self._num_units])

        # state update on the top_list
        s_same = state[:, 1:, :, :]
        s_top_list = state[:, :1, :, :]
        s_top_list_pushed = s_top_list[:, :, :-1, :]
        s_top_list_pushed = tf.concat([x_expanded, s_top_list_pushed], axis=2)
        s_state = tf.concat([s_top_list_pushed, s_same], axis=1)

        # push new list
        s_list_pushed = state[:, :-1, :, :]
        s_top_list = tf.zeros_like(state[:, :1, :-1, :], dtype=tf.float32)
        s_top_list = tf.concat([x_expanded, s_top_list], axis=2)
        s_list = tf.concat([s_top_list, s_list_pushed], axis=1)

        new_s = tf.einsum("bo, bndh -> bndh", p_state, s_state) + tf.einsum("bo, bndh -> bndh", p_list, s_list)
        new_s += tf.einsum("bo, bndh -> bndh", p_no, state)

        new_s = tf.reshape(new_s, [batch_size, -1])
        return new_s, new_s

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "kernel_initializer": initializers.serialize(self._kernel_initializer),
            "bias_initializer": initializers.serialize(self._bias_initializer),
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(NestedListOperationCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def action_layer(encoded_state_seq, hp, name="action_layer"):
    with tf.variable_scope(name):
        action_seq = tf.nn.softmax(tf.layers.dense(encoded_state_seq, 3), axis=-1)
    action_seq = tf.identity(action_seq, "action_seq")
    return action_seq


def action_layer_no_leak(encoded_state_seq, hp, name="action_layer"):
    with tf.variable_scope(name):
        action_seq = tf.nn.softmax(tf.layers.dense(encoded_state_seq, 2), axis=-1)
    action_seq = tf.concat([action_seq, tf.zeros_like(action_seq[:, :, :, :1])], axis=-1)
    action_seq = tf.identity(action_seq, "action_seq")
    return action_seq


def nested_list_operations(input_seq, action_seq, hp, name=""):

    batch_size, _, _, hidden_size = common_layers.shape_list(input_seq)
    # hidden_size = hp.hidden_size
    # if hp.concat_context:
    #     hidden_size *= 2
    cell = NestedListOperationCell(hidden_size,
                                   list_size=hp.list_size,
                                   num_lists=hp.num_lists)

    sequence_length = common_layers.length_from_embedding(input_seq)
    sequence_length = tf.identity(sequence_length, "sequence_length")
    # hidden_size = common_layers.shape_list(input_seq)[-1]

    cell_input = tf.concat([action_seq, input_seq], axis=-1)
    # batch_size = common_layers.shape_list(cell_input)[0]
    cell_input = tf.squeeze(cell_input, axis=2)
    cell_input = tf.identity(cell_input, "cell_input")
    initial_state = tf.zeros(shape=[batch_size, cell.state_size], dtype=tf.float32)
    with tf.variable_scope(name):
        history, final_states = tf.nn.dynamic_rnn(
            cell,
            cell_input,
            sequence_length,
            initial_state=initial_state,
            dtype=tf.float32,
            time_major=False)
    grid_structured_states = tf.reshape(final_states, [-1, cell.num_lists, cell.list_size, hidden_size])
    grid_structured_states = tf.identity(grid_structured_states, "grid_structured_states")
    return grid_structured_states


# transform a sequence to a grid
# input_seq: [batch_size, seq_len, 1, hidden_size]
# action_seq: [batch_size,  seq_len, 1, 1]
# grid_structured_states: [batch_size, num_lists, list_size, hidden_size]

def seq2grid_module_manual(input_seq, action_seq, hp):
    name = "seq2grid"
    with tf.variable_scope(name):
        grid_structured_states = nested_list_operations(input_seq, action_seq, hp)
    return action_seq, grid_structured_states


def seq2grid_module(input_seq, encoded_state_seq, hp, scope_name="seq2grid"):
    with tf.variable_scope(scope_name):
        if hp.get("no_leak", False):
            action_seq = action_layer_no_leak(encoded_state_seq, hp)
        else:
            action_seq = action_layer(encoded_state_seq, hp)
        grid_structured_states = nested_list_operations(input_seq, action_seq, hp)
    return action_seq, grid_structured_states

