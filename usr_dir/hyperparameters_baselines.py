from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_tiny
from tensor2tensor.models.lstm import lstm_attention
from tensor2tensor.models.research.universal_transformer import universal_transformer_tiny

"""
We use transformer_tiny, adaptive_universal_transformer_tiny hyperparameters 
which are provided by tensor2tensor library as default. 
Following hyperparameters are used while training LSTM, LSTM Attention and RMC.
"""

@registry.register_hparams
def lstm_attention_standard_structure_lr3():
	hparams = lstm_attention()
	hparams.learning_rate_constant = 0.001
	hparams.learning_rate_schedule = "constant"
	hparams.num_hidden_layers = 3
	hparams.hidden_size = 64
	hparams.add_hparam("eval_throttle_seconds", 100)
	return hparams

@registry.register_hparams
def lstm_attention_think_lr3():
	hparams = lstm_attention()
	hparams.learning_rate_constant = 0.001
	hparams.learning_rate_schedule = "constant"
	hparams.num_hidden_layers = 1
	hparams.hidden_size = 512
	hparams.add_hparam("eval_throttle_seconds", 100)
	return hparams


@registry.register_hparams
def lstm_attention_standard_structure_lr3_h512():
	hparams = lstm_attention_standard_structure_lr3()
	hparams.hidden_size = 512
	return hparams

@registry.register_hparams
def lstm_attention_standard_structure_lr3_h1024():
	hparams = lstm_attention_standard_structure_lr3()
	hparams.hidden_size = 1024
	return hparams



