from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensor2tensor.layers import common_layers
from tensor2tensor.models import resnet

def dot_product_local_atten_2d(grid_structured, weight_q, weight_k, weight_v, rel_pos_embs):

    _, num_stacks, stack_size, hidden_size = common_layers.shape_list(grid_structured)
    grid_q = tf.nn.conv2d(grid_structured, weight_q, strides=[1, 1, 1, 1], padding='VALID')
    grid_k = tf.nn.conv2d(grid_structured, weight_k, strides=[1, 1, 1, 1], padding='VALID')
    grid_v = tf.nn.conv2d(grid_structured, weight_v, strides=[1, 1, 1, 1], padding='VALID')

    def translate_grid(grid_structured, row_offset, col_offset, constant_values=0):
        padded_grid = tf.pad(grid_structured, [[0, 0], col_offset, row_offset, [0, 0]], constant_values=constant_values)
        padded_grid = padded_grid[:, col_offset[1]:col_offset[1]+num_stacks, row_offset[1]:row_offset[1]+stack_size, :]
        return padded_grid

    def get_bias(grid_structured, row_offset, col_offset):
        return translate_grid(tf.zeros_like(grid_structured), row_offset, col_offset, constant_values=-100000)

    padded_grid_qks = []
    padded_grid_vs = []
    for r, row_offset in enumerate([[1, 0], [0, 0], [0, 1]]):
        for c, col_offset in enumerate([[1, 0], [0, 0], [0, 1]]):
            extended_padded_grid_v = tf.expand_dims(translate_grid(grid_v, row_offset, col_offset), axis=-1)
            padded_grid_vs.append(extended_padded_grid_v)  # [b, n, s, h, 1]

            padded_grid_k = translate_grid(grid_k, row_offset, col_offset)

            padded_grid_qk = tf.reduce_sum(grid_q * padded_grid_k, axis=-1, keep_dims=True)  # [b, n, s, 1]
            padded_grid_qk /= tf.math.sqrt(tf.cast(hidden_size, tf.float32))

            # relative position embedding
            padded_grid_qk += tf.nn.conv2d(grid_q, rel_pos_embs[r*3+c], strides=[1, 1, 1, 1], padding='VALID')

            # masking interactions happened beyond the grid
            bias = get_bias(padded_grid_qk, row_offset, col_offset)
            bias = tf.identity(bias, f"bias_{r}{c}")
            padded_grid_qk += bias

            padded_grid_qks.append(padded_grid_qk)

    padded_grid_qks = tf.concat(padded_grid_qks, axis=-1)  # [batch_size, num_stacks, stack_size, 9]
    padded_grid_qks = tf.nn.softmax(padded_grid_qks, axis=-1)
    padded_grid_qks = tf.identity(padded_grid_qks, "padded_grid_qks_probs")

    padded_grid_vs = tf.concat(padded_grid_vs, axis=-1)  # [batch_size, num_stacks, stack_size, hidden_size, 9]

    self_atten_res = tf.einsum("bnsl, bnshl -> bnsh", padded_grid_qks, padded_grid_vs)
    return self_atten_res


def prepare_local_weights(hp):
    hidden_size = hp.hidden_size
    weight_q = tf.get_variable("w_q", shape=[1, 1, hidden_size, hidden_size], dtype=tf.float32)
    weight_k = tf.get_variable("w_k", shape=[1, 1, hidden_size, hidden_size], dtype=tf.float32)
    weight_v = tf.get_variable("w_v", shape=[1, 1, hidden_size, hidden_size], dtype=tf.float32)

    rel_row_pos_embs = []
    rel_col_pos_embs = []
    for r in range(-1, 2, 1):
        rel_row_pos_embs.append(tf.get_variable(f"rel_row_pos_emb_{r}",
                                                shape=[1, 1, hidden_size//2, 1], dtype=tf.float32))
        rel_col_pos_embs.append(tf.get_variable(f"rel_col_pos_emb_{r}",
                                                shape=[1, 1, hidden_size//2, 1], dtype=tf.float32))

    rel_pos_embs = []
    for r, row_offset in enumerate([[1, 0], [0, 0], [0, 1]]):
        for c, col_offset in enumerate([[1, 0], [0, 0], [0, 1]]):
            rel_pos_embs.append(tf.concat([rel_row_pos_embs[r], rel_col_pos_embs[c]], axis=-2))
    return weight_q, weight_k, weight_v, rel_pos_embs


def local_self_attention(grid_structured, hp, name=""):
    with tf.variable_scope(name):
        weight_q, weight_k, weight_v, rel_pos_embs = prepare_local_weights(hp)
        dot_product_res = dot_product_local_atten_2d(grid_structured, weight_q, weight_k, weight_v, rel_pos_embs)
    return dot_product_res


def bottleneck_tlsa_block(grid_structured, hp):
    data_format = "channels_last"
    is_training = hp.mode == tf.estimator.ModeKeys.TRAIN
    hidden_size_base = hp.hidden_size
    filters_out = 4 * hidden_size_base

    def projection_shortcut(inputs):
        inputs = resnet.conv2d_fixed_padding(inputs, filters_out, kernel_size=1, data_format=data_format,
                                             strides=1, is_training=is_training)
        return resnet.batch_norm_relu(inputs, is_training, relu=False, data_format=data_format)
    residual = projection_shortcut(grid_structured)

    inputs = resnet.conv2d_fixed_padding(
        inputs=grid_structured,
        filters=hidden_size_base,
        kernel_size=1,
        strides=1,
        data_format=data_format,
        is_training=is_training)
    inputs = resnet.batch_norm_relu(inputs, is_training, data_format=data_format)

    inputs = local_self_attention(inputs, hp, name="tlsa")
    inputs = resnet.batch_norm_relu(inputs, is_training, data_format=data_format)

    inputs = resnet.conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=1,
        data_format=data_format,
        is_training=is_training)
    inputs = resnet.batch_norm_relu(
        inputs,
        is_training,
        relu=False,
        init_zero=False,
        data_format=data_format)
    return tf.nn.relu(inputs + residual)


def bottleneck_resnet_block_downsample(grid_structured, hp):
    data_format = "channels_last"
    is_training = hp.mode == tf.estimator.ModeKeys.TRAIN
    hidden_size_base = hp.hidden_size
    filters_out = 4 * hidden_size_base
    strides = 2   # downsample
    use_td = hp.use_td
    targeting_rate = hp.targeting_rate
    keep_prob = hp.keep_prob

    def projection_shortcut(inputs):
        """Project identity branch."""
        inputs = resnet.conv2d_fixed_padding(
            inputs=inputs,
            filters=filters_out,
            kernel_size=1,
            strides=strides,
            data_format=data_format,
            use_td=use_td,
            targeting_rate=targeting_rate,
            keep_prob=keep_prob,
            is_training=is_training)
        return resnet.batch_norm_relu(inputs, is_training, relu=False, data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = resnet.bottleneck_block(
        grid_structured,
        hidden_size_base,
        is_training,
        projection_shortcut,
        strides,
        False,
        data_format,
        use_td=use_td,
        targeting_rate=targeting_rate,
        keep_prob=keep_prob)
    return inputs


def bottleneck_resnet_block_1d(grid_structured, hp):
    data_format = "channels_last"
    is_training = hp.mode == tf.estimator.ModeKeys.TRAIN
    hidden_size_base = hp.hidden_size
    filters_out = 4 * hidden_size_base
    def projection_shortcut(inputs):
        inputs = resnet.conv2d_fixed_padding(inputs, filters_out, kernel_size=1, data_format=data_format,
                                             strides=1, is_training=is_training)
        return resnet.batch_norm_relu(inputs, is_training, relu=False, data_format=data_format)
    residual = projection_shortcut(grid_structured)

    inputs = resnet.conv2d_fixed_padding(
        inputs=grid_structured,
        filters=hidden_size_base,
        kernel_size=1,
        strides=1,
        data_format=data_format,
        is_training=is_training)
    inputs = resnet.batch_norm_relu(inputs, is_training, data_format=data_format)

    inputs = resnet.conv2d_fixed_padding(
        inputs=inputs,
        filters=hidden_size_base,
        kernel_size=[1, 3],
        strides=1,
        data_format=data_format,
        is_training=is_training)
    inputs = resnet.batch_norm_relu(inputs, is_training, data_format=data_format)

    inputs = resnet.conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=1,
        data_format=data_format,
        is_training=is_training)
    inputs = resnet.batch_norm_relu(
        inputs,
        is_training,
        relu=False,
        init_zero=False,
        data_format=data_format)
    return tf.nn.relu(inputs + residual)



def bottleneck_resnet_block(grid_structured, hp, kernel_size=3, out_multiple=4):
    data_format = "channels_last"
    is_training = hp.mode == tf.estimator.ModeKeys.TRAIN
    hidden_size_base = hp.hidden_size
    filters_out = out_multiple * hidden_size_base
    def projection_shortcut(inputs):
        inputs = resnet.conv2d_fixed_padding(inputs, filters_out, kernel_size=1, data_format=data_format,
                                             strides=1, is_training=is_training)
        return resnet.batch_norm_relu(inputs, is_training, relu=False, data_format=data_format)
    residual = projection_shortcut(grid_structured)

    inputs = resnet.conv2d_fixed_padding(
        inputs=grid_structured,
        filters=hidden_size_base,
        kernel_size=1,
        strides=1,
        data_format=data_format,
        is_training=is_training)
    inputs = resnet.batch_norm_relu(inputs, is_training, data_format=data_format)

    inputs = resnet.conv2d_fixed_padding(
        inputs=inputs,
        filters=hidden_size_base,
        kernel_size=kernel_size,
        strides=1,
        data_format=data_format,
        is_training=is_training)
    inputs = resnet.batch_norm_relu(inputs, is_training, data_format=data_format)

    inputs = resnet.conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=1,
        data_format=data_format,
        is_training=is_training)
    inputs = resnet.batch_norm_relu(
        inputs,
        is_training,
        relu=False,
        init_zero=False,
        data_format=data_format)
    return tf.nn.relu(inputs + residual)


def bottleneck_tlsa(grid_structured, hp):
    for layer in range(hp.num_hidden_layers):
        with tf.variable_scope(f"bottleneck_tlsa_block_{layer}"):
            grid_structured = bottleneck_tlsa_block(grid_structured, hp)
    return grid_structured


def bottleneck_resnet(grid_structured, hp, kernel_size=3):
    for layer in range(hp.num_hidden_layers):
        with tf.variable_scope(f"bottleneck_resnet_block_{layer}"):
            grid_structured = bottleneck_resnet_block(grid_structured, hp, kernel_size)
    return grid_structured


def text_tcnn_body(grid_structured_states, hparams):
    """TextCNN main model_fn.
    Args:
      "grid_structured_states": Text inputs.
          [batch_size, num_stacks, stack_size, hidden_dim].
    Returns:
      Final encoder representation. [batch_size, 1, 1, hidden_dim]
    """
    inputs = grid_structured_states

    xshape = common_layers.shape_list(inputs)

    vocab_size = xshape[3]

    pooled_outputs = []
    for _, filter_size in enumerate(hparams.filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, filter_size, vocab_size, hparams.num_filters]
            filter_var = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            filter_bias = tf.Variable(
                tf.constant(0.1, shape=[hparams.num_filters]), name="b")
            conv = tf.nn.conv2d(
                inputs,
                filter_var,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            conv_outputs = tf.nn.relu(
                tf.nn.bias_add(conv, filter_bias), name="relu")
            pooled = tf.math.reduce_max(
                conv_outputs, axis=1, keepdims=True, name="max")
            pooled = tf.math.reduce_max(
                pooled, axis=2, keepdims=True, name="max")
            pooled_outputs.append(pooled)

    num_filters_total = hparams.num_filters * len(hparams.filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add dropout
    output = tf.nn.dropout(h_pool_flat, 1 - hparams.output_dropout)
    output = tf.reshape(output, [-1, 1, 1, num_filters_total])

    return output


def decode_by_decoder_type(grid_structured_states, hp, features=None):
    if hp.decoder_type == "cnn":
        grid_structured_outputs = bottleneck_resnet(grid_structured_states, hp)
    elif hp.decoder_type == "acnn":
        grid_structured_outputs = bottleneck_tlsa(grid_structured_states, hp)
    elif hp.decoder_type == "text_tcnn":
        grid_structured_outputs = text_tcnn_body(grid_structured_states, hp)
    else:
        grid_structured_outputs = grid_structured_states
    grid_structured_outputs = tf.identity(grid_structured_outputs, "grid_structured_outputs")
    return grid_structured_outputs

