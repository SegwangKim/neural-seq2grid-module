from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry, t2t_model
import tensorflow as tf
from . import encoders_for_seq2grid_module
from . import grid_decoders
from . import seq2grid_module


def compute_CE_loss(reversed_padded_target_raw, grid_output_top_list):
    return tf.losses.sparse_softmax_cross_entropy(reversed_padded_target_raw, grid_output_top_list)


def extract_top_list(grid_output, list_size):
    total_len = common_layers.shape_list(grid_output)[1] * common_layers.shape_list(grid_output)[2]
    output_vocab_size = common_layers.shape_list(grid_output)[-1]
    flatten_output = tf.reshape(grid_output, [-1, total_len, 1, output_vocab_size])
    flatten_output = tf.expand_dims(flatten_output, -2)
    top_list = flatten_output[:, :list_size, :, :, :]
    top_list = tf.identity(top_list, "top_list")
    return top_list


def get_argmaxed_top_list(top_list, list_size):
    argmaxed_top_list = tf.argmax(top_list, axis=-1)
    argmaxed_top_list = tf.reshape(argmaxed_top_list, [-1, list_size, 1, 1])
    return argmaxed_top_list


def infer_valid_length_from_top_list(top_list, list_size):
    argmaxed_top_list = get_argmaxed_top_list(top_list, list_size)
    valid_length = common_layers.length_from_embedding(argmaxed_top_list)
    valid_length = tf.identity(valid_length, "output_valid_length")
    return valid_length


def infer_flipped_argmaxed_outputs(top_list, list_size):
    valid_length = infer_valid_length_from_top_list(top_list, list_size)
    argmaxed_top_list = get_argmaxed_top_list(top_list, list_size)
    argmaxed_top_list = tf.identity(argmaxed_top_list, "argmaxed_top_list")
    final = tf.reverse_sequence(argmaxed_top_list, valid_length, seq_axis=1)
    return final


def infer_flipped_outputs(top_list, list_size):
    valid_length = infer_valid_length_from_top_list(top_list, list_size)
    final = tf.reverse_sequence(top_list, valid_length, seq_axis=1)
    return final


def target_reversing_and_padding(target_seq, list_size):
    targets_length = common_layers.length_from_embedding(target_seq)
    flipped_target_seq = tf.reverse_sequence(target_seq, targets_length, seq_axis=1)
    flipped_target_seq = tf.pad(flipped_target_seq, [[0, 0], [0, list_size], [0, 0], [0, 0]])
    flipped_target_seq = flipped_target_seq[:, :list_size, :, :]
    flipped_target_seq = tf.identity(flipped_target_seq, "flipped_target_seq")
    return flipped_target_seq



@registry.register_model
class SeqInGridOutArchitecture(t2t_model.T2TModel):

    def __init__(self, *args, **kwargs):
        super(SeqInGridOutArchitecture, self).__init__(*args, **kwargs)
        self._additional_loss = {}

    def forward_path(self, features):
        input_seq = features["inputs"]
        target_space = features.get("target_space_id", 0)
        hp = self._hparams

        encoded_state_seq = encoders_for_seq2grid_module.encode_input_seq(input_seq, hp,
                                                                          target_space,
                                                                          features)

        actions_seq, grid_structured_states = seq2grid_module.seq2grid_module(input_seq,
                                                                              encoded_state_seq,
                                                                              hp)

        grid_structured_outputs = grid_decoders.decode_by_decoder_type(grid_structured_states, hp, features)
        grid_output_top_list = extract_top_list(grid_structured_outputs, hp.list_size)
        return actions_seq, grid_output_top_list

    def body(self, features):
        target_seq = features["targets_raw"]
        target_seq = tf.identity(target_seq, "target_seq")
        hp = self._hparams

        actions_seq, grid_output_top_list = self.forward_path(features)
        # argmaxed_grid_output = tf.identity(tf.argmax(grid_output_top_list, axis=-1), "argmaxed_grid_output")

        # Match the output to the target modality (shape, dimension)
        output_vocab_size = self._problem_hparams.vocab_size["targets"]
        logits = common_layers.dense(grid_output_top_list, output_vocab_size)

        # Compute loss
        self.compute_loss(target_seq, logits)

        # Flip the logits for evaluation during training
        final = infer_flipped_outputs(logits, hp.list_size)
        return {"targets": final}, self._additional_loss

    def compute_loss(self, target_seq, logits):
        list_size = self._hparams.list_size
        flipped_target_seq = target_reversing_and_padding(target_seq, list_size)
        loss_CE = compute_CE_loss(flipped_target_seq, logits)
        self._additional_loss["training"] = loss_CE

    def _greedy_infer(self, features, decode_length, use_tpu=False):
        with tf.variable_scope(self.name):
            features = self.bottom(features)
            with tf.variable_scope("body"):
                actions_seq, grid_output_top_list = self.forward_path(features)
                output_vocab_size = self._problem_hparams.vocab_size["targets"]
                logits = common_layers.dense(grid_output_top_list, output_vocab_size)
            final = infer_flipped_argmaxed_outputs(logits, self._hparams.list_size)
            return final


@registry.register_model
class SeqInGridOutArchitectureGuided(SeqInGridOutArchitecture):
    def forward_path(self, features):
        input_seq = features["inputs"]
        hp = self._hparams
        actions_seq = tf.expand_dims(features["action"], axis=-2)
        actions_seq = tf.identity(actions_seq, "action_seq")
        actions_seq, grid_structured_states = seq2grid_module.seq2grid_module_manual(input_seq,
                                                                                     actions_seq,
                                                                                     hp)

        grid_structured_outputs = grid_decoders.decode_by_decoder_type(grid_structured_states, hp, features)
        grid_output_top_list = extract_top_list(grid_structured_outputs, hp.list_size)
        return actions_seq, grid_output_top_list


@registry.register_model
class SeqInGridOutToLabelManual(t2t_model.T2TModel):
    def __init__(self, *args, **kwargs):
        super(SeqInGridOutToLabelManual, self).__init__(*args, **kwargs)
        self._additional_loss = {}

    def body(self, features):
        _, out = self.forward_path(features)
        out = tf.reduce_mean(out, [1, 2], keepdims=True)

        return out

    def forward_path(self, features):
        input_seq = features["inputs"]
        hp = self._hparams
        actions_seq = tf.expand_dims(features["action"], axis=-2)
        actions_seq = tf.identity(actions_seq, "action_seq")
        actions_seq, grid_structured_states = seq2grid_module.seq2grid_module_manual(input_seq,
                                                                                     actions_seq,
                                                                                     hp)
        logit_for_label = grid_decoders.decode_by_decoder_type(grid_structured_states, hp, features)
        return actions_seq, logit_for_label


    def infer(self,
              features=None,
              decode_length=50,
              beam_size=1,
              top_beams=1,
              alpha=0.0,
              use_tpu=False):
        """Predict."""
        del decode_length, beam_size, top_beams, alpha, use_tpu
        assert features is not None
        logits, _ = self(features)  # pylint: disable=not-callable
        assert len(logits.get_shape()) == 5
        logits = tf.squeeze(logits, [1, 2, 3])
        log_probs = common_layers.log_prob_from_logits(logits)
        predictions, scores = common_layers.argmax_with_score(log_probs)
        return {
            "outputs": predictions,
            "scores": scores,
        }


@registry.register_model
class SeqInGridOutToLabel(SeqInGridOutToLabelManual):
    def forward_path(self, features):
        input_seq = features["inputs"]
        target_space = features.get("target_space_id", 0)
        hp = self._hparams

        encoded_state_seq = encoders_for_seq2grid_module.encode_input_seq(input_seq, hp,
                                                                          target_space,
                                                                          features)

        actions_seq, grid_structured_states = seq2grid_module.seq2grid_module(input_seq,
                                                                              encoded_state_seq,
                                                                              hp)

        logit_for_label = grid_decoders.decode_by_decoder_type(grid_structured_states, hp, features)
        return actions_seq, logit_for_label
