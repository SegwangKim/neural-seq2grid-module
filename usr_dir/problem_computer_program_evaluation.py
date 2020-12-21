from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems, text_encoder
from tensor2tensor.utils import registry
import numpy as np
from random import choice, random, randint

TOTAL_VARIALBES = [i for i in "abcdefghij"]

class ExprAndValue:
    def __init__(self, expr, value):
        self.value = value
        self.expr = expr

    def __str__(self):
        return "expr:{}, value:{}".format(self.expr, self.value)


def get_operand(length, stack):
    if len(stack) == 0:
        value = randint(1, 10 ** length-1)
        expr = str(value)
        return ExprAndValue(expr, value)
    k = stack.pop()
    return k


def get_operands(length, nr, stack):
    ret = {}
    perm = np.random.permutation([i for i in range(nr)])
    for i in perm:
        ret[i] = get_operand(length, stack)
    ret = sorted([(k, v) for k, v in ret.items()], key=lambda e: e[0])
    return [v for (k, v) in ret]


def unused_var(used_vars):
    unseen_var = choice(list(set(TOTAL_VARIALBES) - set(used_vars)))
    used_vars.append(unseen_var)
    return unseen_var


def pair_opr(length, stack, used_vars):
    ve1, ve2 = get_operands(length, 2, stack)
    if random() < 0.5:
        value = ve1.value + ve2.value
        expr = "({}+{})".format(ve1.expr, ve2.expr)
    else:
        value = ve1.value - ve2.value
        expr = "({}-{})".format(ve1.expr, ve2.expr)
    return [], expr, value


def smallmul_opr(length, stack, used_vars):
    ve1 = get_operand(length, stack)
    b = randint(1, 4 * length)
    value = ve1.value * b
    if random() < 0.5:
        expr = "({}*{})".format(ve1.expr, b)
    else:
        expr = "({}*{})".format(b, ve1.expr)
    return [], expr, value


def equality_opr(length, stack, used_vars):
    ve1 = get_operand(length, stack)
    return [], ve1.expr, ve1.value


def vars_opr(length, stack, used_vars):
    var = unused_var(used_vars)
    ve1, ve2 = get_operands(length, 2, stack)
    if random() < 0.5:
        value = ve1.value + ve2.value
        code = ["{}={}".format(var, ve1.expr)]
        expr = "({}+{})".format(var, ve2.expr)
    else:
        value = ve1.value - ve2.value
        code = ["{}={}".format(var, ve1.expr)]
        expr = "({}-{})".format(var, ve2.expr)
    return code, expr, value


def small_loop_opr(length, stack, used_vars):
    var = unused_var(used_vars)
    ve1, ve2 = get_operands(length, 2, stack)
    loop = randint(1, 4 * length)
    if random() < 0.5:
        op = "+"
        value = ve1.value + loop * ve2.value
    else:
        op = "-"
        value = ve1.value - loop * ve2.value
    code = ["{}={}".format(var, ve1.expr),
            "for x in range({}):{}{}={}".format(loop, var, op, ve2.expr, var)]
    return code, var, value


def ifstat_opr(length, stack, used_vars):
    ve1, ve2, ve3, ve4 = get_operands(length, 4, stack)
    if random() < 0.5:
        name = ">"
        if ve1.value > ve2.value:
            value = ve3.value
        else:
            value = ve4.value
    else:
        name = "<"
        if ve1.value < ve2.value:
            value = ve3.value
        else:
            value = ve4.value
    expr = "({} if {}{}{} else {})".format(ve3.expr, ve1.expr, name, ve2.expr, ve4.expr)
    return [], expr, value


def generate_program(length, nesting, mixed):
    stack = []
    used_vars = []
    funcs = [pair_opr, smallmul_opr, equality_opr, vars_opr, small_loop_opr, ifstat_opr]
    code = []
    nesting = randint(1, nesting) if mixed else nesting
    length = randint(1, length) if mixed else length
    for h in range(nesting):
        f = choice(funcs)
        code_tmp, var_tmp, output_tmp = f(length, stack, used_vars)
        code += code_tmp
        stack.append(ExprAndValue(var_tmp, output_tmp))
    final_ve = stack.pop()
    enc = "\n".join(code) + "\n" if len(code) > 0 else ""
    enc += "print(" + final_ve.expr + ")"
    dec = final_ve.value
    return enc, str(dec)


@registry.register_problem
class ComputerProgramEvaluation(text_problems.Text2TextProblem):
    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        return text_encoder.ByteTextEncoder()

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

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

    @property
    def is_generate_per_split(self):
        return True

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        num_data = int(1e6)

        def gen_seq(shard, residue, mixed):
            length, nesting = self.shard_spec(shard)
            while 1:
                enc, dec = generate_program(length, nesting, mixed)
                if hash(enc) % 3 == residue:
                    return {"inputs": enc, "targets": dec}

        if dataset_split == problem.DatasetSplit.TRAIN:
            for num in range(num_data):
                yield gen_seq(0, 0, True)

        elif dataset_split == problem.DatasetSplit.EVAL:
            for num in range(500):
                yield gen_seq(0, 1, True)

        else:
            for num in range(10000):
                for shard in range(self.dataset_splits[-1]["shards"]):
                    yield gen_seq(shard, 2, False)

    def shard_spec(self, shard):
        if shard == 0:
            length, nesting = 5, 2
        elif shard == 1:
            length, nesting = 7, 2
        return length, nesting
