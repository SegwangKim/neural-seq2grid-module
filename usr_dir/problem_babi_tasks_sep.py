# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import babi_qa
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry
import os, shutil
import tensorflow as tf


_DIR_NAME = "tasks_1-20_v1-2"
_TAR = _DIR_NAME + ".tar.gz"
_URL = "http://www.thespermwhale.com/jaseweston/babi/" + _TAR

_TASKS = {
    "qa0": "qa0_all-tasks",
    "qa1": "qa1_single-supporting-fact",
    "qa2": "qa2_two-supporting-facts",
    "qa3": "qa3_three-supporting-facts",
    "qa4": "qa4_two-arg-relations",
    "qa5": "qa5_three-arg-relations",
    "qa6": "qa6_yes-no-questions",
    "qa7": "qa7_counting",
    "qa8": "qa8_lists-sets",
    "qa9": "qa9_simple-negation",
    "qa10": "qa10_indefinite-knowledge",
    "qa11": "qa11_basic-coreference",
    "qa12": "qa12_conjunction",
    "qa13": "qa13_compound-coreference",
    "qa14": "qa14_time-reasoning",
    "qa15": "qa15_basic-deduction",
    "qa16": "qa16_basic-induction",
    "qa17": "qa17_positional-reasoning",
    "qa18": "qa18_size-reasoning",
    "qa19": "qa19_path-finding",
    "qa20": "qa20_agents-motivations"
}

# A list of problem names that are registered by this module. This will get
# populated at module load time in the code at the bottom of this file.
REGISTERED_PROBLEMS = []

MAX_SENTENCE_LENGTH = 15
TEMP = MAX_SENTENCE_LENGTH - 1


def babi_parser_v2(tmp_dir,
                   babi_task_id,
                   subset,
                   dataset_split,
                   joint_training=True):

    def _data_file(mode, task_id):
        file_name = (_TASKS[task_id] + "_{}.txt")
        return os.path.join(_DIR_NAME, subset, file_name.format(mode))

    def _all_task_raw_data_generator(tmp_dir, data_file, dataset_split):
        tf.logging.info("Preparing dataset of all task together")
        globe_name = ("*_{}.txt")
        mode_name = "test"
        if dataset_split == problem.DatasetSplit.TRAIN:
            mode_name = "train"
        files_name = os.path.join(
            tmp_dir, _DIR_NAME, subset,
            globe_name.format(mode_name))
        with tf.gfile.GFile(data_file, "wb") as outfile:
            for filename in tf.gfile.Glob(files_name):
                if filename == data_file:
                    # don"t want to copy the output into the output
                    continue
                with tf.gfile.GFile(filename, "rb") as readfile:
                    shutil.copyfileobj(readfile, outfile)

    def _parse_answer(answer):
        if (joint_training or babi_task_id in ["qa8", "qa19", "qa0"
                                               ]):  # "lists-sets" or "path finding"
            return "".join([d for d in answer.split(",")])  # as a single token!
        else:
            return answer

    if dataset_split == problem.DatasetSplit.TRAIN:
        babi_train_task_id = "qa0" if joint_training else babi_task_id
        data_file = os.path.join(tmp_dir, _data_file("train", babi_train_task_id))
    else:
        data_file = os.path.join(tmp_dir, _data_file("test", babi_task_id))

    if ((babi_task_id == "qa0" or joint_training) and
            not tf.gfile.Exists(os.path.join(tmp_dir, data_file))):
        _all_task_raw_data_generator(tmp_dir, data_file, dataset_split)

    tf.logging.info("Parsing %s into training/testing instances...", data_file)

    babi_instances = []
    with tf.gfile.GFile(data_file, mode="r") as f:
        story = []
        story_idx = []
        for k, line in enumerate(f):
            line_splitted = line.strip().split(" ")
            line_num, line = line_splitted[0], " ".join(line_splitted[1:])
            if int(line_num) == 1:
                story = []
                story_idx = []
            if "\t" in line:
                question, answer, supporting_sentence_nums = line.split("\t")
                question = babi_qa._normalize_string(question)
                substories = [s for s in story if s]
                answer = _parse_answer(answer)
                supporting_sens = [idx for idx, i in enumerate(story_idx)
                                   if i in supporting_sentence_nums.split()]
                instance = {
                    babi_qa.FeatureNames.STORY: substories,
                    babi_qa.FeatureNames.QUESTION: question,
                    babi_qa.FeatureNames.ANSWER: answer,
                    "supporting_sens": supporting_sens,
                }

                babi_instances.append(instance)
                story.append("")
            else:
                story.append(babi_qa._normalize_string(line))
                story_idx.append(line_num)
    return babi_instances


@registry.register_problem
class BabiQaClsSepAllTasks_10k(babi_qa.BabiQaConcat):

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
            "shards": 21,
        }]

    def dataset_filename(self):
        return self.name

    @property
    def joint_training(self):
        return True

    @property
    def babi_task_id(self):
        return "qa0"

    @property
    def babi_subset(self):
        return "en-10k"

    def generate_samples(self, data_dir, tmp_dir, dataset_split):

        # tmp_dir = babi_qa._prepare_babi_data(tmp_dir, data_dir)
        babi_qa._build_vocab(
            self.generate_text_for_vocab(data_dir, tmp_dir), data_dir,
            self.vocab_filename)

        if dataset_split == problem.DatasetSplit.TEST:
            all_examples = []

            for babi_task_id in ["qa"+str(i) for i in range(0, 21)]:
                examples = babi_parser_v2(tmp_dir, babi_task_id, self.babi_subset,
                                          dataset_split, self.joint_training)
                all_examples.append(examples)

            def _generate_samples():
                for num in range(1000):
                    for examples in all_examples:
                        example = examples[num]
                        context, inputs, context_action, targets, supporting_sens = self.concat_with_sep(example)
                        action = [0] * len(inputs.split(" ")[:-1]) + [2] + context_action
                        inputs = inputs + " " + context
                        assert len(action) == len(inputs.split(" "))
                        yield {
                            "inputs": inputs,
                            "targets": targets,
                            "action": action,
                            "supporting_sens": supporting_sens,
                        }

            return _generate_samples()

        else:
            examples = babi_parser_v2(tmp_dir, self.babi_task_id, self.babi_subset,
                                      dataset_split, self.joint_training)

            def _generate_samples():
                for example in examples:
                    context, inputs, context_action, targets, supporting_sens = self.concat_with_sep(example)
                    action = [2] + [0] * len(inputs.split(" ")[1:-1]) + [2] + context_action
                    #        <CLS> +                  q               + <EOQ> + ...
                    inputs = inputs + " " + context
                    assert len(action) == len(inputs.split(" "))
                    yield {
                        "inputs": inputs,
                        "targets": targets,
                        "action": action,
                        "supporting_sens": supporting_sens,
                    }
            return _generate_samples()


    def generate_text_for_vocab(self, data_dir, tmp_dir):
        # NOTE: for babi, we create the vocab from both train and test data.
        for dataset_split in [problem.DatasetSplit.TRAIN, problem.DatasetSplit.EVAL]:

            for example in babi_qa._babi_parser(tmp_dir, self.babi_task_id, self.babi_subset,
                                                dataset_split, self.joint_training):

                context = " ".join(example[babi_qa.FeatureNames.STORY])

                yield " ".join(context.split())
                yield "<CLS> " + " ".join(example[babi_qa.FeatureNames.QUESTION].split()) + " <EOQ>"
                yield example[babi_qa.FeatureNames.ANSWER]

    def concat_with_sep(self, example):
        context = " ".join(example[babi_qa.FeatureNames.STORY])
        context = context.split(".")
        context = [sent.strip() for sent in context if len(sent.strip()) > 0]

        context = [f"{sent} ." for sent in context]

        context_action = []
        for i, sent in enumerate(context):
            if i in example["supporting_sens"]:
                context_action += [1] + [0] * (len(sent.split(" ")) - 1)
            else:
                context_action += [2] * len(sent.split(" "))
            if i == len(context) - 1:
                context[i] = f"{sent} <EOS>"
                context_action += [2]
        context = " ".join(context)

        prev_input = " ".join(example[babi_qa.FeatureNames.QUESTION].split())
        new_input = "<CLS> " + prev_input + " <EOQ>"
        return context, new_input, context_action, example[babi_qa.FeatureNames.ANSWER], example["supporting_sens"]

    def hparams(self, defaults, unused_model_hparams):
        super(BabiQaClsSepAllTasks_10k, self).hparams(defaults, unused_model_hparams)
        p = defaults
        if "context" in p.modality:
            del p.modality["context"]
        if "context" in p.vocab_size:
            del p.vocab_size["context"]
        p.modality["action"] = modalities.ModalityType.IDENTITY
        p.vocab_size["action"] = 3

    def feature_encoders(self, data_dir):
        encoders = (super(BabiQaClsSepAllTasks_10k, self).feature_encoders(data_dir))
        encoders["action"] = text_encoder.RealEncoder()
        return encoders

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        label_encoder = self.get_labels_encoder(data_dir)
        for sample in generator:
            inputs = encoder.encode(sample["inputs"])
            targets = label_encoder.encode(sample["targets"])
            sample["targets"] = targets
            yield {"inputs": inputs,
                   "targets": targets,
                   "action": sample["action"],
                   "supporting_sens": sample["supporting_sens"]}

    def example_reading_spec(self):
        data_fields, data_items_to_decoders = (super(BabiQaClsSepAllTasks_10k, self).example_reading_spec())
        data_fields["supporting_sens"] = tf.VarLenFeature(tf.int64)
        data_fields["action"] = tf.VarLenFeature(tf.int64)
        return (data_fields, data_items_to_decoders)

    def preprocess_example(self, example, mode, hparams):
        example["action"] = tf.one_hot(example["action"], depth=3, dtype=tf.float32)
        return example


    def eval_metrics(self):
        return [
            metrics.Metrics.ACC,
            metrics.Metrics.ACC_PER_SEQ,
        ]

    @property
    def vocab_filename(self):
        return "vocab_sep.%s.%s" % (self.dataset_filename(), text_problems.VocabType.TOKEN)
