import tensorflow as tf
from copy import deepcopy
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensor2tensor.utils import trainer_lib, hparams_lib
from tensor2tensor.layers import common_hparams

def show_dict(dic, sess, cutoff_size=100, squeeze=True):
    dic_np = sess.run(dic)
    for k, v in dic_np.items():
        print(k, np.shape(v))
        if len(np.shape(v)) > 0 and np.cumprod(np.shape(v))[-1] > cutoff_size:
            continue
        if "loss" in k:
            print(v)
            continue
        if squeeze:
            print(np.squeeze(v))
            continue
        else:
            print(v)
    return dic_np


def keyword_filter(keywords, op, end_check=True):
    if isinstance(keywords, str):
        keywords = [keywords]
    if end_check:
        names = op.name.split("/")
        for keyword in keywords:
            if keyword in names[-1]:
                return True
        return False
    else:
        for keyword in keywords:
            if keyword in op.name:
                return True
        return False


def get_char_img():
    image = mpimg.imread("./results/character_images/sample_chars.png")
    image = image[15:-43, 3:-2, 0]
    chars = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    char_image = {}
    unit = 118
    for idx, char in enumerate(chars):
        row = idx // 9
        col = idx % 9
        char_image[char] = image[unit * row:unit*(row+1), unit * col:unit*(col+1)]
    preprocessed_char_image = {}
    for char, img in char_image.items():
        preprocessed_char_image[char] = img > 0
    preprocessed_char_image[" "] = 0.5 * np.ones_like(preprocessed_char_image["1"])
    return preprocessed_char_image, unit

def print_argmax_actions(action_probs, input_tokens, encoders):
    print(instant_decode(encoders, input_tokens, strip_at_eos=False, line_break=False))
    marker = ""
    for idx in range(len(input_tokens)):
        if "Byte" not in str(encoders["inputs"]):
            now = instant_decode(encoders, input_tokens[idx:idx+1], strip_at_eos=False)
            now = str(np.argmax(np.squeeze(action_probs[idx]), axis=-1)) + re.sub("\S", "_", now[1:])
            marker += now+ " "
        else:
            now = str(np.squeeze(np.argmax(action_probs[idx], axis=-1)))
            marker += now
    print(marker)


def path_set_up(model_name, hparam_set, problem_name, data_dir, sess_dir, global_steps=None):
    data_dir = os.path.expanduser(data_dir)
    train_dir = os.path.expanduser(sess_dir+problem_name+'-'+model_name+'-'+hparam_set)
    ckp = train_dir+"/model.ckpt-"+str(global_steps) if global_steps else train_dir
    return data_dir, train_dir, ckp


def hparams_set_up(problem_name, data_dir, hparam_set=None, hparams_override=None):
    if hparam_set:
        hparams = trainer_lib.create_hparams(hparam_set, hparams_overrides_str=hparams_override)
    else:
        hparams = common_hparams.basic_params1()
    hparams.data_dir = data_dir
    hparams_lib.add_problem_hparams(hparams, problem_name)
    return hparams, hparams.problem


def model_hp_dict(problem_name, basic_dir, model_name=None, hp_name=None):
    model_hp = {}
    if isinstance(model_name, str):
        model_name = [model_name]
    if isinstance(hp_name, str):
        hp_name = [hp_name]
    for i in os.listdir(basic_dir):
        problem_i, model_i, hp_i = i.split("-")
        if problem_name != problem_i:
            continue
        if model_name != None and model_i not in model_name:
            continue
        if hp_name != None and hp_i not in hp_name:
            continue
        if model_hp.get(model_i, -1) == -1:
            model_hp[model_i] = [hp_i]
        else:
            model_hp[model_i].append(hp_i)
    return model_hp

# Setup helper functions for encoding and decoding
def instant_encode(encoders, input_str, append_eos=True):
    """Input str to features dict, ready for inference"""
    if len(re.findall("[0-9]", input_str)) == 0:
        inputs = encoders["inputs"].encode(input_str)
        if append_eos:
            inputs = inputs + [1]
    else:
        inputs = [encoders["inputs"].encode(i)[0] for i in input_str]
        if append_eos:
            inputs = inputs + [1]
    batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
    return {"inputs": batch_inputs}


def instant_decode(encoders, integers, modality="inputs", strip_at_eos=True, line_break=True):
    """List of ints to str"""
    if len(np.squeeze(integers).shape) == 0:
        integers = np.expand_dims(np.squeeze(integers), 0)
    else:
        integers = list(np.squeeze(integers))
    if 1 in integers and strip_at_eos:
        one_place = [idx for idx, i in enumerate(integers) if i == 1]
        integers = integers[:one_place[-1]]
    if not line_break:
        return encoders[modality].decode(integers).replace("\n", "~")
    return encoders[modality].decode(integers)


def print_float_matrix(matrix):
    # matrix = matrix.cpu().data.numpy()
    for row in matrix:
        print(" ".join([f"{j:.2f}" for j in row]))


def print_int_matrix(matrix):
    # matrix = matrix.cpu().data.numpy()
    for row in matrix:
        print(" ".join([f"{j}" if j != 0 else "_" for j in row]))


def print_chunks(raw_cmd, ops_for_cmd, field):
    res = []
    cmd = field.iltos(raw_cmd, delimiter=" ")
    chunk_cmd = ""
    for op, cm in zip(ops_for_cmd, cmd.split(" ")):
        if op == 1:
            res.append([0, chunk_cmd])
            res = [[chunk[0]+1, chunk[1]] for chunk in res]
            chunk_cmd = ""
        chunk_cmd += f"{cm} "
    res.append([0, chunk_cmd])
    for i in res[::-1]:
        print(i)


def call_html():
    import IPython
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))


def dense2onehot(dense_op, vocab_size=3):
    dense_op = np.array(dense_op)
    onehot_op = np.zeros((dense_op.size, vocab_size))
    onehot_op[np.arange(dense_op.size), dense_op] = 1
    return onehot_op


def op_tlu(img, token):
    vocab_size = img.shape[-1]
    new_elt = dense2onehot(token, vocab_size)
    ans = deepcopy(img)
    ans[0] = np.array(list(new_elt) + list(ans[0, :-1]))
    return ans


def op_nlp(img, token):
    num_lists, list_size, vocab_size = img.shape
    top_list = np.zeros_like(img[0])
    new_elt = dense2onehot(token, vocab_size)
    top_list = np.array(list(new_elt) + list(top_list[:-1]))
    ans = deepcopy(img)
    ans[1:] = ans[:-1]
    ans[0] = top_list
    return ans


def evolution(img, token, prob):
    img_tlu = prob[0] * op_tlu(img, token)
    img_nlp = prob[1] * op_nlp(img, token)
    img_no = prob[2] * img
    return img_tlu + img_nlp + img_no


def emulate_NLA_by_one_hots(tokens, probs, list_size, num_lists, vocab_size, steps=-1):
    total_len = np.sum([i != 0 for i in np.squeeze(tokens)])
    probs = np.squeeze(probs)[:total_len]
    probs = probs[:steps + 1] if steps >= 0 else probs

    img = np.zeros((num_lists, list_size, vocab_size))
    for idx, prob in enumerate(probs):
        img = evolution(img, tokens[idx], prob)
        # str = ""
        # for i in img:
        #     for j in i:
        #         str += f"{np.argmax(j):02d} "
        #     str += "\n"
        # print(str)
    return img


def emulate_NLA_by_step(tokens, probs, list_size, num_lists, steps=-1):
    total_len = np.sum([i != 0 for i in np.squeeze(tokens)])
    probs = np.squeeze(probs)[:total_len]
    probs = probs[:steps + 1] if steps >= 0 else probs

    step_grid = np.zeros((num_lists, list_size, total_len))
    for idx, prob in enumerate(probs):
        step_grid = evolution(step_grid, idx, prob)
    return step_grid


def step_grid_to_symbol_grid(step_grid, tokens, vocab_size, stepwise_max=True):
    num_lists, list_size, steps = np.shape(step_grid)
    symbol_grid = np.zeros((num_lists, list_size, vocab_size))
    for step in range(steps):
        grid = step_grid[:, :, step]
        if stepwise_max:
            ind = np.unravel_index(np.argmax(grid, axis=None), grid.shape)
            symbol_grid[ind[0], ind[1], tokens[step]] = np.max(grid)
        else:
            for h, row in enumerate(grid):
                for w, weight in enumerate(row):
                    symbol_grid[h, w, tokens[step]] += weight
                    # print(h, w, tokens[step], weight, symbol_grid[h, w, tokens[step]])
    return symbol_grid


def img_symbols(img_onehot, encoders, top_k=None, allowable_indice=None):
    char_image, unit = get_char_img()
    num_lists, list_size, vocab_size = img_onehot.shape
    final_image = np.zeros([unit * num_lists, unit * list_size])
    for row, row_list in enumerate(img_onehot):
        for col, weights in enumerate(row_list):
            if top_k:
                sorted_weights = sorted([[idx, weight] for idx, weight in enumerate(weights)], key=lambda x: -x[1])
                for top in range(min(top_k, len(sorted_weights))):
                    idx, weight = sorted_weights[top]
                    decoded_token = encoders["inputs"].decode([idx])
                    final_image[row*unit:(row+1)*unit, col*unit:(col+1)*unit] += char_image.get(decoded_token[0],
                                                                                                char_image["~"]) * weight
            else:
                for idx, weight in enumerate(weights):
                    decoded_token = encoders["inputs"].decode([idx])
                    final_image[row*unit:(row+1)*unit, col*unit:(col+1)*unit] += char_image.get(decoded_token[0],
                                                                                                char_image["~"]) * weight
    return final_image


def plot_attention(attention_map, input_tags=None, output_tags=None):
    attn_len = len(attention_map)

    # Plot the attention_map
    plt.clf()
    f = plt.figure(figsize=(15, 10))
    ax = f.add_subplot(1, 1, 1)

    # Add image
    i = ax.imshow(attention_map, interpolation='nearest', cmap='Blues')

    # Add colorbar
    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)

    # Add labels
    ax.set_yticks(range(attn_len))
    if output_tags != None:
      ax.set_yticklabels(output_tags[:attn_len])

    ax.set_xticks(range(attn_len))
    if input_tags != None:
      ax.set_xticklabels(input_tags[:attn_len], rotation=45)
    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')
    # add grid and legend
    ax.grid()
    plt.show()

