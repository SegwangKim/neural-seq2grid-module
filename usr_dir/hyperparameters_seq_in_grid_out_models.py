from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
from tensor2tensor.utils import registry
from tensor2tensor.models.lstm import lstm_seq2seq
from tensor2tensor.layers import common_hparams

HIDDEN_SIZE = 64
LIST_SIZE = 25
NUM_LISTS = 3
ENCODER_TYPE = 'gru'
NUM_HIDDEN_LAYERS = 3
DECODER_TYPE = "cnn"


# sigo stands for sequence-input-grid-output
def sigo_hp_basic():
    hparams = lstm_seq2seq()
    hparams.max_target_seq_length = 0
    hparams.max_input_seq_length = 0
    hparams.batch_size = 1024
    hparams.learning_rate_constant = 0.001
    hparams.learning_rate_schedule = "constant"
    hparams.dropout = 0.2
    hparams.num_hidden_layers = NUM_HIDDEN_LAYERS
    hparams.initializer = "uniform_unit_scaling"
    hparams.clip_grad_norm = 2.0
    hparams.hidden_size = HIDDEN_SIZE
    hparams.add_hparam("encoder_type", ENCODER_TYPE)
    hparams.add_hparam("decoder_type", DECODER_TYPE)
    hparams.add_hparam("list_size", LIST_SIZE)
    hparams.add_hparam("num_lists", NUM_LISTS)
    hparams.add_hparam("filter_sizes", [2, 3, 4, 5])
    hparams.add_hparam("num_filters", 128)

    # Targeted dropout for RESNET
    hparams.add_hparam("use_td", False)
    hparams.add_hparam("targeting_rate", None)
    hparams.add_hparam("keep_prob", None)


    # for transformer encoder
    hparams.add_hparam("proximity_bias", False)
    hparams.add_hparam("pos", "timing")  # timing, none
    hparams.add_hparam("num_encoder_layers", 0)
    hparams.add_hparam("num_decoder_layers", 0)
    hparams.add_hparam("attention_dropout", 0.0)
    hparams.add_hparam("num_heads", 4)
    hparams.add_hparam("use_pad_remover", True)
    hparams.add_hparam("attention_key_channels", 0)
    hparams.add_hparam("attention_value_channels", 0)
    hparams.add_hparam("self_attention_type", "dot_product")
    hparams.add_hparam("ffn_layer", "dense_relu_dense")
    hparams.add_hparam("filter_size", 64)
    hparams.add_hparam("relu_dropout", 0.0)
    hparams.add_hparam("relu_dropout_broadcast_dims", "")

    return hparams


def text_cnn_2d_base():
    """Set of hyperparameters."""
    hparams = common_hparams.basic_params1()
    hparams.batch_size = 4096
    hparams.max_length = 256
    hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
    hparams.optimizer_adam_epsilon = 1e-9
    hparams.learning_rate_schedule = "legacy"
    hparams.learning_rate_decay_scheme = "noam"
    hparams.learning_rate = 0.1
    hparams.learning_rate_warmup_steps = 4000
    hparams.initializer_gain = 1.0
    hparams.num_hidden_layers = 6
    hparams.initializer = "uniform_unit_scaling"
    hparams.weight_decay = 0.0
    hparams.optimizer_adam_beta1 = 0.9
    hparams.optimizer_adam_beta2 = 0.98
    hparams.num_sampled_classes = 0
    hparams.label_smoothing = 0.1
    hparams.shared_embedding_and_softmax_weights = True
    hparams.symbol_modality_num_shards = 16

    # Add new ones like this.
    hparams.add_hparam("filter_sizes", [2, 3, 4])
    hparams.add_hparam("num_filters", 128)
    hparams.add_hparam("output_dropout", 0.4)

    # sigo_hparams
    hparams.add_hparam("encoder_type", ENCODER_TYPE)
    hparams.add_hparam("decoder_type", DECODER_TYPE)
    hparams.add_hparam("list_size", LIST_SIZE)
    hparams.add_hparam("num_lists", NUM_LISTS)

    hparams.add_hparam("pos", "timing")  # timing, none
    hparams.add_hparam("attention_dropout", 0.0)
    hparams.add_hparam("num_heads", 4)
    return hparams


def sigo_hparam(**kwargs):
    if kwargs.get("legacy", None) or kwargs.get("decoder_type", None) == "TEXT_TCNN":
        hparams = text_cnn_2d_base()
    else:
        hparams = sigo_hp_basic()

    if 'encoder_type' in kwargs.keys():
        hparams.encoder_type = kwargs['encoder_type'].lower()
    if 'decoder_type' in kwargs.keys():
        hparams.decoder_type = kwargs['decoder_type'].lower()
    if 'hidden_size' in kwargs.keys():
        hparams.hidden_size = kwargs['hidden_size']
    if 'num_hidden_layers' in kwargs.keys():
        hparams.num_hidden_layers = kwargs['num_hidden_layers']
    if 'num_lists' in kwargs.keys():
        hparams.num_lists = kwargs['num_lists']
    if 'list_size' in kwargs.keys():
        hparams.list_size = kwargs['list_size']

    return hparams


def hp_spec2prefix(**kwargs):
    encoder_type = kwargs.get('encoder_type', ENCODER_TYPE)
    decoder_type = kwargs.get('decoder_type', DECODER_TYPE)  # number-level-cnn too long, omit
    hidden_size = kwargs.get('hidden_size', HIDDEN_SIZE)
    num_layers = kwargs.get("num_hidden_layers", NUM_HIDDEN_LAYERS)

    prefix = f"H{hidden_size}L{num_layers}{encoder_type}_{decoder_type}"
    if 'num_lists' in kwargs.keys() and 'list_size' in kwargs.keys():
        num_lists = kwargs['num_lists']
        list_size = kwargs['list_size']
        prefix += f"_{num_lists}x{list_size}"
    return prefix+f""


def _hps_to_register():
    # [encoder_type, attention_mechanism, concat_context, list_size, num_lists, nsc_lambda]
    hp_specs = [
        {'hidden_size': 128, 'num_hidden_layers': 3, 'encoder_type': 'MLP', 'decoder_type': 'CNN'},
        {'hidden_size': 128, 'num_hidden_layers': 3, 'encoder_type': 'TE', 'decoder_type': 'CNN'},
        {'hidden_size': 128, 'num_hidden_layers': 3, 'encoder_type': 'GRU', 'decoder_type': 'CNN'},
        {'hidden_size': 128, 'num_hidden_layers': 3, 'encoder_type': 'GRU', 'decoder_type': 'ACNN'},
        {'hidden_size': 128, 'num_hidden_layers': 2, 'encoder_type': 'GRU', 'decoder_type': 'TEXT_TCNN',
         "num_lists": 4, "list_size": 8},
    ]
    hp_setups = dict([(hp_spec2prefix(**hp_spec), hp_spec) for hp_spec in hp_specs])
    return hp_setups


def hp_generator(prefix, hp_spec):
    hp_fn = lambda: sigo_hparam(**hp_spec)
    hp_fn.__name__ = sigo_hparam.__name__ + prefix
    return hp_fn


def _register_hps():
    for prefix, hp_setup in six.iteritems(_hps_to_register()):
        registry.register_hparams(hp_generator(prefix, hp_setup))


_register_hps()

