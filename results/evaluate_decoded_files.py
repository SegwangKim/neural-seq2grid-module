import tensorflow as tf
import numpy as np
import os, re
import pandas as pd
import getpass
from IPython.display import display

SIZE = 60
USER = getpass.getuser()
DECODE_DIR_NAME = "decode_00000"


def acc_per_seq(output_lines, target_lines, tokens=False):
    res = 0.0
    num = len(target_lines)
    if tokens:
        for output_line, target_line in zip(output_lines, target_lines):
            output_tokens = output_line.split()
            target_tokens = target_line.split()
            try:
                if target_tokens == output_tokens[:len(target_tokens)]:
                    res += 1.0
            except:
                continue
        res = res / num
        return res

    for output_line, target_line in zip(output_lines, target_lines):
        if output_line == target_line:
            res += 1.0
    res = res / num   
    return res


def acc_per_char(output_lines, target_lines, reverse=False, tokens=False):
    res = 0.0
    if tokens:
        if reverse:
            num = sum([len(output_line.split()) for output_line in output_lines])
            for output_line, target_line in zip(output_lines, target_lines):
                for idx, target_sym in enumerate(target_line.split()):
                    try:
                        if target_sym == output_line.split()[idx]:
                            res += 1.0
                    except:
                        continue
            res = res / num if num > 0 else 0
        else:
            num = sum([len(target_line.split()) for target_line in target_lines])
            for output_line, target_line in zip(output_lines, target_lines):
                for idx, output_sym in enumerate(output_line.split()):
                    try:
                        if output_sym == target_line.split()[idx]:
                            res += 1.0
                    except:
                        continue
            res = res / num if num > 0 else 0
        return res

    if reverse:
        num = sum([len(output_line) for output_line in output_lines])
        for output_line, target_line in zip(output_lines, target_lines):
            for idx, target_sym in enumerate(target_line):
                try:
                    if target_sym == output_line[idx]:
                        res += 1.0
                except:
                    continue
        res = res / num if num > 0 else 0
    else:
        num = sum([len(target_line) for target_line in target_lines])
        for output_line, target_line in zip(output_lines, target_lines):
            for idx, output_sym in enumerate(output_line):
                try:
                    if output_sym == target_line[idx]:
                        res += 1.0
                except:
                    continue
        res = res / num if num > 0 else 0
    return res


def gather_model_hp_pairs_by_problem(basic_path, problem, model=None, hp=None, trial=None):
    model_hp_trial = []
    if isinstance(model, str):
        model = [model]
    if isinstance(hp, str):
        hp = [hp]
    for i in os.listdir(basic_path):
        if len(re.findall("-", i)) != 2:
            continue
        problem_i, model_i, hp_i = i.split("-")
        try:
            hp_i, trial_i = hp_i.split(".")
        except:
            continue
        trial_i = int(trial_i)
        if problem != problem_i:
            continue
        if model != None and model_i not in model:
            continue
        if hp != None and hp_i not in hp:
            continue
        if trial != None and trial_i not in trial:
            continue
        model_hp_trial.append([model_i, hp_i, trial_i])
    return model_hp_trial


def get_ckpt(output_dir, global_steps=None):
    ckpt = os.path.join(output_dir, f"model.ckpt-{global_steps}") if global_steps else output_dir
    return ckpt


def read_decoded_files(output_dir, global_steps, file, shard):
    """Score each line in a file and return the scores."""
    decode_dir = os.path.join(output_dir, "decode_00000")
    keyword = f"{global_steps}{file}{shard:03d}"
    decodes_filename, inputs_filename, targets_filename = "", "", ""
    for filename_ in os.listdir(decode_dir):
        if len(re.findall(keyword, filename_)) == 0:
            continue
        if len(re.findall("decodes", filename_)) > 0:
            decodes_filename = filename_
        if len(re.findall("inputs", filename_)) > 0:
            inputs_filename = filename_
        if len(re.findall("targets", filename_)) > 0:
            targets_filename = filename_

    def read_lines(decode_dir, filename):
        if len(filename) == 0:
            print(f"{keyword} decoded files no exist.")
            return []
        with tf.gfile.Open(os.path.join(decode_dir, filename)) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        return lines

    output_lines = read_lines(decode_dir, decodes_filename)
    input_lines = read_lines(decode_dir, inputs_filename)
    target_lines = read_lines(decode_dir, targets_filename)

    # Assume each line corresponds to an instance. So, computer program evaluation problem is not the case.
    input_lines, output_lines, target_lines = remove_empty_lines(input_lines, output_lines, target_lines)

    return input_lines, output_lines, target_lines


def get_step_file_shard(output_dir):
    triplets = []
    if os.path.exists(output_dir) and DECODE_DIR_NAME in os.listdir(output_dir):
        decodes_files = os.listdir(os.path.join(output_dir, DECODE_DIR_NAME))
        for decode_file in decodes_files:
            if len(re.findall('^[0-9]+[a-z]+[0-9]+', decode_file)) == 0:
                continue
            stepfileshard = re.findall('^[0-9]+[a-z]+[0-9]+', decode_file)[0]
            a, b = re.search("[a-z]+", stepfileshard).start(), re.search("[a-z]+", stepfileshard).end()
            step, file, shard = stepfileshard[:a], stepfileshard[a:b], stepfileshard[b:]
            triplets.append((step, file, int(shard)))
        return list(set(triplets))
    else:
        print(f"NOT EXIST: {output_dir}")
    return []


def compose_output_dir(basic_path, problem, model, hp, trial):
    output_dir = os.path.join(basic_path, "-".join([problem, model, f"{hp}.{trial:02d}"]))
    return output_dir


def gather_all_statistics_by_problem(basic_pathes, problem, model, hp, token,
                                     do_over=False):
    df_all = []
    for basic_path in basic_pathes:
        df = get_statistics_on_basic_path(basic_path, problem, model, hp, token, do_over)
        df_all.append(df)
    df_all = pd.concat(df_all, ignore_index=True)
    return df_all


def get_statistics_on_basic_path(basic_path, problem, model=None, hp=None, tokens=False, do_over=False):
    path_name = basic_path
    model_hp_trials = gather_model_hp_pairs_by_problem(basic_path=basic_path, problem=problem, model=model, hp=hp)
    df = get_statistics(basic_path, problem, model_hp_trials, tokens, do_over)
    if len(df) == 0:
        return
    df['directory'] = path_name
    return df


def remove_empty_lines(input_lines, output_lines, target_lines):
    valid_line_num = -1
    for idx, line in enumerate(input_lines):
        if line.split(" ")[0] == "<pad>":
            valid_line_num = idx
            break
    if valid_line_num > 0:
        return input_lines[:valid_line_num], output_lines[:valid_line_num], target_lines[:valid_line_num]
    else:
        return input_lines, output_lines, target_lines


def get_statistics(basic_path, problem, model_hp_trials, tokens=False, do_over=False):
    column_names = ["problem_name", "model", "hp", "trial", "steps", "file", "shard",
                    "avg_len", "seq_acc", "seq_acc_rev", "ch_acc", "ch_acc_rev"]

    def get_statistics_file_path(output_dir, global_steps, file, shard):
        statistics_file = os.path.join(output_dir, DECODE_DIR_NAME,
                                       f"statistics-{global_steps}{file}{int(shard):03d}.pkl")
        return statistics_file

    df = pd.DataFrame(columns=column_names)
    for model_hp_trial in model_hp_trials:
        model, hp, trial = model_hp_trial
        output_dir = compose_output_dir(basic_path, problem, model, hp, trial)
        step_file_shard_triplets = get_step_file_shard(output_dir)
        for triplet in step_file_shard_triplets:
            global_steps, file, shard = triplet
            statistics_file_path = get_statistics_file_path(output_dir, global_steps, file, shard)
            if os.path.exists(statistics_file_path) and not do_over:
                df_temp = pd.read_pickle(statistics_file_path)
            else:
                input_lines, output_lines, target_lines = read_decoded_files(output_dir,
                                                                             global_steps,
                                                                             file,
                                                                             shard)
                a = acc_per_seq(output_lines, target_lines, tokens=tokens)
                aa = acc_per_seq(target_lines, output_lines, tokens=tokens)
                b = acc_per_char(output_lines, target_lines, tokens=tokens)
                c = acc_per_char(output_lines, target_lines, reverse=True, tokens=tokens)
                ratio = np.average([len(oline) / len(tline) for oline, tline in zip(output_lines, target_lines)])
                res = [problem, model, hp, trial, global_steps, file, int(shard), ratio, a, aa, b, c]
                df_temp = pd.DataFrame([res], columns=column_names)
                df_temp.to_pickle(statistics_file_path)
            df = pd.concat([df, df_temp], ignore_index=True)
    return df


def read_decoded_files_by_df_index(df, idx):
    a = df.iloc[idx]
    output_dir = compose_output_dir(a.directory, a.problem_name, a.model, a.hp, int(a.trial))
    input_lines, output_lines, target_lines = read_decoded_files(output_dir,
                                                                 a.steps,
                                                                 a.file,
                                                                 a.shard)
    return {"src": input_lines, "pred": output_lines, "trg": target_lines}
