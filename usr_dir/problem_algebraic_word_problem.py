from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

from mathematics_dataset.util import composition, display
from mathematics_dataset.modules import arithmetic

import random
import math
import sympy

_ADD_SUB_ENTROPY_TRAIN = (4, 16)
_ADD_SUB_ENTROPY_EXTRAPOLATE = (16, 20)


def _entropy_for_pair(sample_args):
    entropy, sample_args = sample_args.peel()
    entropy_p = max(1, random.uniform(0, entropy))
    entropy_q = max(1, entropy - entropy_p)
    return entropy_p, entropy_q, sample_args


def _entropy_for_ood_hard_pair(sample_args):
    low_limit = sample_args.min_entropy
    entropy_p, _ = sample_args.peel()
    entropy_q, sample_args = sample_args.peel()
    assert (entropy_p > low_limit) & (entropy_q > low_limit)
    return entropy_p, entropy_q, sample_args


def integer(entropy, bounded=False, coprime_to=1):
    max_ = math.pow(10, entropy)
    low_bound = 1

    if bounded:
        low_bound_entropy = int(entropy) - 1
        low_bound = math.pow(10, low_bound_entropy)
        max_ = int(math.ceil(max_))
        range_ = [low_bound, max_]
        if random.choice([True, False]):
            range_ = [-max_, -low_bound]
    else:
        max_ = int(math.ceil(max_ / 2))
        range_ = [-max_, max_]
    while True:
        value = random.randint(*range_)
        if abs(value) >= low_bound and sympy.gcd(value, coprime_to) == 1:
            break
    return sympy.Integer(value)



def add_or_sub(entropy, entropy_fn, bounded=False):
    context = None
    is_question = context is None
    context = composition.Context()

    is_addition = random.choice([False, True])
    entropy_p, entropy_q, sample_args = entropy_fn(entropy)

    p = display.Decimal(integer(entropy_p, bounded))
    q = display.Decimal(integer(entropy_q, bounded))
    p, q = context.sample(sample_args, [p, q])

    if is_addition:
        return arithmetic._add_question_or_entity(context, p, q, is_question)
    else:
        return arithmetic._sub_question_or_entity(context, p, q, is_question)


@registry.register_problem
class AlgebraicWordProblem(text_problems.Text2TextProblem):

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

    def shard_spec(self, shard):
        if shard == 0:
            entropy = composition.PreSampleArgs(1, 1, *_ADD_SUB_ENTROPY_TRAIN)
            entropy_fn = _entropy_for_pair
            bounded = False
        else:
            entropy = composition.PreSampleArgs(1, 1, *_ADD_SUB_ENTROPY_EXTRAPOLATE)
            entropy_fn = _entropy_for_ood_hard_pair
            bounded = True
        return entropy, entropy_fn, bounded

    def gen_input_target(self, entropy_generator):
        entropy, entropy_fn, bounded = entropy_generator
        sample = add_or_sub(entropy, entropy_fn, bounded)
        return {
            "inputs": str(sample.question),
            "targets": str(sample.answer),
        }

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir

        if dataset_split == problem.DatasetSplit.TRAIN:
            for num in range(int(1*1e6)):
                yield self.gen_input_target(self.shard_spec(0))
        elif dataset_split == problem.DatasetSplit.EVAL:
            for num in range(int(1*1e4)):
                yield self.gen_input_target(self.shard_spec(0))
        else:
            for num in range(10000):
                for shard in range(self.dataset_splits[-1]["shards"]):
                    yield self.gen_input_target(self.shard_spec(shard))

