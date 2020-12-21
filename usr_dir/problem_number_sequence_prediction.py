from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems, text_encoder
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
import tensorflow as tf
import random


def initial_term_sampler(length, deg, mixed):
    initial_terms = []
    for _ in range(deg):
        if mixed:
            length2 = random.randint(1, length)
            num = random.choice([-1, 1]) * random.randint(10**(length2-1), 10**length2-1)
        else:
            num = random.choice([-1, 1]) * random.randint(1, 10**length-1)
        initial_terms.append(num)
    return initial_terms


def generate_progression(initial_terms, coeffs, num_terms):
    assert len(initial_terms) == len(coeffs)
    assert num_terms > len(initial_terms)
    deg = len(coeffs)
    def recursion_fn(terms, coeffs):
        return sum([t * c for t, c in zip(terms, coeffs)])

    terms = initial_terms
    for _ in range(num_terms - deg):
        next_term = recursion_fn(terms[-deg:], coeffs)
        terms += [next_term]
    return terms


@registry.register_problem
class NumberSequencePredictionThird(text_problems.Text2TextProblem):
    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    @property
    def is_generate_per_split(self):
        return True

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 2,
        }]

    def progression_coefficients(self):
        coeffs = [1, -1, 2]
        return coeffs

    def gen_input_target(self, shard, residue, mixed):
        length, num_terms = self.shard_spec(shard)
        while 1:
            coeffs = self.progression_coefficients()
            deg = len(coeffs)
            initial_terms = initial_term_sampler(length, deg, mixed)
            terms = generate_progression(initial_terms, coeffs, num_terms+1)
            input_terms = terms[:-1]
            target_terms = terms[-1:]
            enc = " ".join([str(t) for t in input_terms])
            dec = " ".join([str(t) for t in target_terms])
            if hash(enc) % 3 == residue:
                return {"inputs": enc, "targets": dec}

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        num_data = int(1e6)

        if dataset_split == problem.DatasetSplit.TRAIN:
            for num in range(num_data):
                yield self.gen_input_target(0, 0, True)

        elif dataset_split == problem.DatasetSplit.EVAL:
            for num in range(500):
                yield self.gen_input_target(0, 1, True)

        else:
            for num in range(10000):
                for shard in range(self.dataset_splits[-1]["shards"]):
                    yield self.gen_input_target(shard, 2, False)


    def shard_spec(self, shard):
        if shard == 0:
            length, num_terms = 4, random.randint(4, 6)
        else:
            length, num_terms = 6, random.randint(10, 12)
        return length, num_terms


@registry.register_problem
class ToyAddition(text_problems.Text2TextProblem):
    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.stop_at_eos = int(True)

        p.modality = {"targets": modalities.ModalityType.SYMBOL}
        p.vocab_size = {"targets": self._encoders["inputs"].vocab_size}
        p.modality["action"] = modalities.ModalityType.IDENTITY
        p.vocab_size["action"] = 3 # No-effect
        if self.has_inputs:
            p.modality["inputs"] = modalities.ModalityType.SYMBOL
            p.vocab_size["inputs"] = self._encoders["inputs"].vocab_size

    @property
    def is_generate_per_split(self):
        return True

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 6,
        }]

    def progression_coefficients(self):
        coeffs = [1, 1]
        return coeffs

    def gen_input_target(self, shard, residue, mixed):
        length, num_terms = self.shard_spec(shard)
        while 1:
            coeffs = self.progression_coefficients()
            deg = len(coeffs)
            initial_terms = initial_term_sampler(length, deg, mixed)
            initial_terms = [abs(i) for i in initial_terms]
            terms = generate_progression(initial_terms, coeffs, num_terms+1)
            input_terms = terms[:-1]
            target_terms = terms[-1:]
            enc = " ".join([str(t) for t in input_terms])
            action = [1 if idx>0 and enc[idx-1] == " " else 0 for idx, i in enumerate(enc)]+[0]
            dec = " ".join([str(t) for t in target_terms])
            if hash(enc) % 3 == residue:
                return {"inputs": enc, "targets": dec, "action": action}

    @property
    def num_train_data(self):
        return int(1e6)

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        num_data = self.num_train_data

        if dataset_split == problem.DatasetSplit.TRAIN:
            for num in range(num_data):
                yield self.gen_input_target(0, 0, True)

        elif dataset_split == problem.DatasetSplit.EVAL:
            for num in range(500):
                yield self.gen_input_target(0, 1, True)

        else:
            for num in range(500):
                for shard in range(1, 1+self.dataset_splits[-1]["shards"]):
                    yield self.gen_input_target(shard, 2, False)


    def shard_spec(self, shard):
        if shard == 0:
            length, num_terms = 5, 2
        elif shard == 1:
            length, num_terms = 3, 2
        elif shard == 2:
            length, num_terms = 4, 2
        elif shard == 3:
            length, num_terms = 5, 2
        elif shard == 4:
            length, num_terms = 6, 2
        elif shard == 5:
            length, num_terms = 7, 2
        else:
            length, num_terms = 8, 2
        return length, num_terms

    def example_reading_spec(self):
        data_fields = {"targets": tf.VarLenFeature(tf.int64)}
        data_fields.update({"action": tf.VarLenFeature(tf.int64)})
        if self.has_inputs:
            data_fields["inputs"] = tf.VarLenFeature(tf.int64)
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def feature_encoders(self, data_dir):
        encoder = self.get_or_create_vocab(data_dir, None, force_get=True)
        return {
            "inputs": encoder,
            "action": text_encoder.RealEncoder(),
            "targets": encoder
        }

    def preprocess_example(self, example, mode, hparams):
        example["action"] = tf.one_hot(example["action"], depth=3, dtype=tf.float32)
        return example


