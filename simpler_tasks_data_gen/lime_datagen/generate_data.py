import pathlib
import shutil
import subprocess
import argparse
import random, string
import tqdm
import copy
import os
import re
import time
import string

# from prototypical_tasks import get_index


def sample_substring(string, min, max):
    substring_len = random.randint(min, max)
    init = random.randint(0, len(string) - substring_len)
    return string[init:init + substring_len], init, substring_len


class Pattern:
    def __init__(self, pattern_list):
        self.pattern_list = pattern_list

    def replace(self, to_replace, replace):
        indices = []
        for i, char in enumerate(self.pattern_list):
            if char == to_replace:
                indices.append(i)

        new_pattern = []
        new_pattern.extend(self.pattern_list[:indices[0]])
        if not isinstance(replace, list):
            replace = [replace]
        new_pattern.extend(replace)
        for i, j in zip(indices[:-1], indices[1:]):
            new_pattern.extend(self.pattern_list[i+1:j])
            new_pattern.extend(replace)
        new_pattern.extend(self.pattern_list[indices[-1]+1:])
        return new_pattern



class SubstRule:
    '''
    example:
    addcomm = SubstRule(["A", "B"], "A+B=B+A")
    out = addcomm.rewrite({"A":"xwd239+c", "B":"23oijs_c"})
    xwd239+c+23oijs_c=23oijs_c+xwd239+c
    '''
    def __init__(self, upper_case_letters, pattern):
        self.upper_case_letters = upper_case_letters
        self.pattern = Pattern(pattern)
    def subst(self, substitute):
        out = self.pattern
        for char in self.upper_case_letters:
            out = Pattern(out.replace(char, substitute[char]))
        return out.pattern_list


def gen_subst(upper_case_letters, lower_case_letters, math_symbols):
    substitute = {}
    substitute_str = []
    for char in upper_case_letters:
        subst_len = random.randint(2, 8)
        subst = random.choices(lower_case_letters + math_symbols,
                               k=subst_len)
        substitute.update({char: subst})
        substitute_str.append(char)
        substitute_str.append(":")
        substitute_str.append("[")
        substitute_str.extend(subst)
        substitute_str.append("]")
        substitute_str.append(",")
    return substitute, substitute_str[:-1]

NON_NUMERIC_SYMBOLS = [':', '<sep>', '<UPPER>', '[', '<LOWER>', '<MATH>', ']']
def gen_one_lime_data(mode, length_range, token_ids, is_original_vocab_division,
                    ida_num_chars_range, ida_pattern_length_range, 
                       num_upper_case_letters, num_lower_case_letters, num_math_symbols):
    vocab = list(token_ids)
    input, output = None, None
    while input is None or input_length > length_range[1] or input_length < length_range[0] or output_length > length_range[1] or output_length < length_range[0]:
        input, output = gen_data(None, None, 1, None, [mode], vocab=vocab, is_original_vocab_division=is_original_vocab_division,
                                ida_num_chars_range=ida_num_chars_range, 
                                ida_pattern_length_range=ida_pattern_length_range, 
                                num_upper_case_letters=num_upper_case_letters, 
                                num_lower_case_letters=num_lower_case_letters, 
                                num_math_symbols=num_math_symbols)
        input_length = len(input.split(' '))
        output_length = len(output.split(' '))
        # print(input_length)
        # breakpoint()
    input = input.replace('space', 'sep').replace(' ,','')
    output = output.replace('space', 'sep').replace(' ,','')
    # input = input[:-1].split(' ')
    # output = output[:-1].split(' ')
    # return input, output
    # print(input)
    # print(input[-1])
    # print(output)
    # print(output[-1])
    # breakpoint()
    return input[:-1], output[:-1] ## because last char is \n


def gen_data(root, name, num, mode_str, modes="subst", vocab=None, is_original_vocab_division=False,
            ida_num_chars_range=None, ida_pattern_length_range=None, 
            num_upper_case_letters=None, num_lower_case_letters=None, num_math_symbols=None):
    is_return_data_and_dont_write = False
    if root is None:
        is_return_data_and_dont_write = True
        assert num == 1
    if not is_return_data_and_dont_write:
        train_src = open(f'{root}/{mode_str}/{name}.src', 'w')
        train_tgt = open(f'{root}/{mode_str}/{name}.tgt', 'w')
    ra = range(num) if num == 1 else tqdm.tqdm(range(num)) 
    for _ in ra:
        vocab_length = len(vocab)
        if is_original_vocab_division:
            if vocab_length >= 68:
                new_vocab = random.sample(vocab, k=68)
                upper_case_letters = new_vocab[:24]
                lower_case_letters = new_vocab[24:48]
                math_symbols = new_vocab[48:68]
            else:
                raise NotImplementedError
        else:
            # new_vocab = vocab
            vocab_division = [num_upper_case_letters, num_lower_case_letters, num_math_symbols]
            is_custom_vocab_division = all([num is not None for num in vocab_division])
            # either all 3 are set or none at all 
            assert is_custom_vocab_division or all([num is None for num in vocab_division])
            print(is_custom_vocab_division)
            if not is_custom_vocab_division:
                vocab_length_d_3 = int(vocab_length/3)
                num_upper_case_letters, num_lower_case_letters, num_math_symbols = vocab_length_d_3, vocab_length_d_3, vocab_length_d_3
                vocab_division = [num_upper_case_letters, num_lower_case_letters, num_math_symbols]

            new_vocab = random.sample(vocab, k=sum(vocab_division))
            upper_case_letters = new_vocab[:num_upper_case_letters]
            lower_case_letters = new_vocab[num_upper_case_letters:num_upper_case_letters+num_lower_case_letters]
            math_symbols = new_vocab[num_upper_case_letters+num_lower_case_letters:]

        mode = random.choice(modes)

        if mode in ["rewrite", "induct_rewrite"]:
            seq_len = random.randint(10, 20)
            lhs = random.choices(math_symbols, k=round(seq_len * 1.5)) + \
                    random.choices(lower_case_letters, k=seq_len)
            random.shuffle(lhs)
            lhs_pattern, init, substring_len = sample_substring(lhs, 5, 20)

            letters = []
            for v in lhs_pattern:
                if v in lower_case_letters:
                    letters.append(v)
            letters = set(letters)
            upper_case_letters = upper_case_letters[:len(letters)]
            substitute = {c: l for c, l in zip(upper_case_letters, letters)}
            for char, letter in substitute.items():
                lhs_pattern = Pattern(lhs_pattern).replace(letter, char)
            rhs_pattern_len = random.randint(5, 20)
            rhs_pattern = random.choices(math_symbols, k=rhs_pattern_len) + list(upper_case_letters)
            random.shuffle(rhs_pattern)
            rule = SubstRule(upper_case_letters, rhs_pattern)
            rhs_substring = rule.subst(substitute)
            rhs = lhs[:init] + rhs_substring + lhs[init + substring_len:]
            pattern = lhs_pattern + ["->"] + rhs_pattern
            if not is_return_data_and_dont_write:
                train_src.write(" ".join([str(c) for c in ["<UPPER>"] + upper_case_letters])+ " ")
                train_src.write(" ".join([str(c) for c in ["<LOWER>"] + lower_case_letters])+ " ")
                train_src.write(" ".join([str(c) for c in ["<MATH>"] + math_symbols])+ " <space> ")
            if mode == "induct_rewrite":
                if not is_return_data_and_dont_write:
                    train_src.write(" ".join([str(c) for c in lhs + ["<space>"] + rhs]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in pattern]) + '\n')
            elif mode == "rewrite":
                if not is_return_data_and_dont_write:
                    train_src.write(" ".join([str(c) for c in pattern + ["<space>"] + lhs]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in rhs]) + '\n')


        elif mode in ["induct_hard", "induct_v2", "induct", "deduct", "abduct", "induct_v2_double", "induct_v3"]:
            if ida_num_chars_range is None:
                num_chars = random.randint(3, 5)
            else:
                num_chars = random.randint(ida_num_chars_range[0], ida_num_chars_range[1])
            if ida_pattern_length_range is None:
                pattern_length = random.choice(range(5, 20))
            else:
                pattern_length = random.randint(ida_pattern_length_range[0], ida_pattern_length_range[1])
            upper_case_letters = upper_case_letters[:num_chars]
            pattern_chars = random.choices(upper_case_letters, k=num_chars * 2)
            upper_case_letters = list(set(pattern_chars))
            pattern = pattern_chars+random.choices(math_symbols, k=pattern_length)
            random.shuffle(pattern)
            rule = SubstRule(upper_case_letters, pattern)
            substitute, substitute_str = gen_subst(upper_case_letters, lower_case_letters, math_symbols)
            result = rule.subst(substitute)
            substitute_2, substitute_str_2 = gen_subst(upper_case_letters, lower_case_letters, math_symbols)
            result_2 = rule.subst(substitute_2)
            # breakpoint()
            tmp1 = " ".join([str(c) for c in ["<UPPER>"] + upper_case_letters]) + " "
            tmp2 = " ".join([str(c) for c in ["<LOWER>"] + lower_case_letters]) + " "
            tmp3 = " ".join([str(c) for c in ["<MATH>"] + math_symbols]) + " <space> "
            input = tmp1 + tmp2 + tmp3
            if not is_return_data_and_dont_write:
                train_src.write(" ".join([str(c) for c in ["<UPPER>"] + upper_case_letters]) + " ")
                train_src.write(" ".join([str(c) for c in ["<LOWER>"] + lower_case_letters]) + " ")
                train_src.write(" ".join([str(c) for c in ["<MATH>"] + math_symbols]) + " <space> ")
            if mode == "induct_hard":
                if not is_return_data_and_dont_write:
                    train_src.write(" ".join([str(c) for c in result]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in pattern]) + '\n')
            elif mode == "induct_v2":
                if not is_return_data_and_dont_write:
                    train_src.write(" ".join([str(c) for c in result]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in substitute_str + ["<space>"] + pattern]) + '\n')
            elif mode == "induct":
                if not is_return_data_and_dont_write:
                    train_src.write(" ".join([str(c) for c in result + ["<space>"] + substitute_str]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in pattern]) + '\n')
                input = input + " ".join([str(c) for c in result + ["<space>"] + substitute_str]) + '\n'
                output = " ".join([str(c) for c in pattern]) + '\n'
            elif mode == "induct_v2_double":
                if not is_return_data_and_dont_write:
                    train_src.write(" ".join([str(c) for c in result+["<space>"]+result_2]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in substitute_str + ["<space>"] +substitute_str_2 +["<space>"]+ pattern]) + '\n')
            elif mode == "induct_v3":
                if not is_return_data_and_dont_write:
                    train_src.write(" ".join([str(c) for c in result+["<space>"]+result_2]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in pattern]) + '\n')
            elif mode == "deduct":
                if not is_return_data_and_dont_write:
                    train_src.write(" ".join([str(c) for c in pattern+ ["<space>"] + substitute_str]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in result]) + '\n')
                input = input + " ".join([str(c) for c in pattern+ ["<space>"] + substitute_str]) + '\n'
                output = " ".join([str(c) for c in result]) + '\n'
            elif mode == "abduct":
                if not is_return_data_and_dont_write:
                    train_src.write(" ".join([str(c) for c in pattern+ ["<space>"] +result]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in substitute_str]) + '\n')
                input = input + " ".join([str(c) for c in pattern+ ["<space>"] +result]) + '\n'
                output = " ".join([str(c) for c in substitute_str]) + '\n'
        else:
            raise ValueError("Mode {} not found".format(mode))
    if is_return_data_and_dont_write:
        return input, output
    train_src.close()
    train_tgt.close()

def main(root, num_train, num_test, mode, vocab_size):
    mode_str = "-".join(mode)
    mode_str += "_vocab{}".format(vocab_size)
    mode_str += "_train{}M".format(num_train/1000000)
    try:
        shutil.rmtree(f'{root}/{mode_str}/')
    except:
        pass
    pathlib.Path(f'{root}/{mode_str}/').mkdir()
    vocab = list(range(vocab_size))

    generate(root=root, num_train=num_train, mode_str=mode_str,
             num_test=num_test, mode=mode, vocab=vocab)

def generate(root, num_train, mode_str, num_test, mode, vocab):
    root_bin = f'{root}/{mode_str}/data-bin/'
    pathlib.Path(root_bin).mkdir(parents=True)

    gen_data(root, 'train', num_train, mode_str, mode, vocab)
    gen_data(root, 'test', num_test, mode_str, mode, vocab)
    gen_data(root, 'valid', num_test, mode_str, mode, vocab)

    command = ['fairseq-preprocess', '--source-lang', 'src', '--target-lang',
               'tgt', '--destdir', root_bin,
               '--trainpref', f'{root}/{mode_str}/train',
               '--validpref', f'{root}/{mode_str}/valid',
               '--testpref', f'{root}/{mode_str}/test',
               '--joined-dictionary'
               ]

    #subprocess.check_call(command)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str,
                        default="data", help="store directory")
    parser.add_argument("--mode", type=str, nargs='+',
                        default=["rewrite"], help="task mode")
    parser.add_argument("--num_train", type=int,
                        default=10000, help="num of train")
    parser.add_argument("--vocab_size", type=int,
                        default=100, help="num of train")
    parser.add_argument("--num_test",
                        type=int,
                        default=1000)
    args = parser.parse_args()

    main(args.root, args.num_train, args.num_test, args.mode, args.vocab_size)
