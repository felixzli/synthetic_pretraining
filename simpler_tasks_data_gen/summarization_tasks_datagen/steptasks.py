from collections import Counter, defaultdict
from typing import List
from base import RealWorldGeneratorForBaseline, BaseGenerator
from overrides import overrides
import numpy as np
import jsonlines
import os
from transformers import T5Tokenizer
from tqdm import tqdm
from copy import deepcopy
from vocab import RandomVocab


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#<<<<<<<<<<<<<<<<<<<<<<< Use this part for using nonsense documents to create pretraining datapoints
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class Modifier():
    def __init__(self,
                 required_tokens):
        self.required_tokens = required_tokens
        self.generator = BaseGenerator(numsent_tolerance = 0, sentlen_tolerance = 5)
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")

    def __next__(self):
        para : List[str] = self.generator.get_para(mean_numsents=100, mean_sentlen=10)
        if para is None:
            return None
        para = [s.lower() for s in para]

        sents = []
        num_toks=0
        valid=False
        for sent in para:
            tokens = self.tokenizer.tokenize(sent)
            num_toks+=len(tokens)
            sents.append(sent)
            if num_toks>self.required_tokens:
                valid=True
                break
        if not valid or (num_toks>self.required_tokens+50): # second condition makes sure the last sentence is not too long
            return None

        article_lines, output_lines = self.modify(sents)
        return {
            "article_lines": article_lines,
            "summary_lines": output_lines
        }

    def modify(self, text):
        raise NotImplementedError

all_tokens_list = RandomVocab().tokens

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#<<<<<<<<<<<<<<<<<<<<<< Use this part for using wikipedia documents to create pretraining datapoints
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# class Modifier():
#     def __init__(self,
#                  required_tokens):
#         self.generator = RealWorldGeneratorForBaseline(required_tokens=required_tokens)
#         self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
#
#     def __next__(self):
#         para : List[str] = self.generator.get_para()
#         if para is None:
#             return None
#         article_lines, output_lines = self.modify(para)
#         return {
#             "article_lines": article_lines,
#             "summary_lines": output_lines
#         }
#
#     def modify(self, text):
#         raise NotImplementedError
#
# all_tokens_list = open("../dataset_root/pretrained_t5_vocabulary/tokens.txt", encoding="utf-8").read().strip().split("\n")

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>




class SentenceReordering(Modifier):
    def __init__(self):
        super().__init__(required_tokens=512)

    @overrides
    def modify(self, para: List[str]):
        output_sents = deepcopy(para)
        input_sents = deepcopy(para)
        np.random.shuffle(input_sents)
        return input_sents, output_sents


class NextSentenceGeneration(Modifier):
    def __init__(self):
        super().__init__(required_tokens=768)
    @overrides
    def modify(self, para: List[str]):
        txt = " ".join(para)
        words = self.tokenizer.tokenize(txt)
        in_length = 512
        out_length = 256
        inp = self.tokenizer.convert_tokens_to_string(words[:in_length])
        outp = self.tokenizer.convert_tokens_to_string(words[in_length:in_length+out_length])
        return [inp], [outp]


class MaskedDocumentGeneration(Modifier):
    def __init__(self):
        super().__init__(required_tokens=512)
        self.token_list = all_tokens_list

    @overrides
    def modify(self, para: List[str]):
        txt = " ".join(para)
        words = self.tokenizer.tokenize(txt)
        masked_len = np.random.randint(100,256)
        start_pos = np.random.randint(512-masked_len+1)

        outp = self.tokenizer.convert_tokens_to_string(words)
        sampled_seq = np.random.rand(masked_len)
        for (_i, samp)  in zip(range(start_pos, start_pos+masked_len), sampled_seq):
            if samp<0.8:
                words[_i] = "<extra_id_0>"
            elif samp<0.9:
                words[_i] = np.random.choice(self.token_list)
            else:
                pass # leave the token as it is
        inp = self.tokenizer.convert_tokens_to_string(words)

        return [inp], [outp]


def write_task_dataset(task_name, trainset, valset, testset):
    if not os.path.exists("../dataset_root/pretraining_datasets"):
        os.mkdir("../dataset_root/pretraining_datasets")
    base_path = f"../dataset_root/pretraining_datasets/{task_name}"

    os.mkdir(base_path)
    for (splitname, split) in zip(["train", "val", "test"], [trainset, valset, testset]):
        with jsonlines.open(os.path.join(base_path, f"{splitname}.jsonl"), "w") as w:
            for dp in split:
                w.write(dp)


if __name__=="__main__":
    TRAIN_SIZE = 100000
    VAL_SIZE = 5000
    TEST_SIZE = 5000
    TOTAL_DATASET_SIZE = TRAIN_SIZE+VAL_SIZE+TEST_SIZE


    ###############################################################

    gen = NextSentenceGeneration()

    train_dataset = []
    hits, misses=0,0
    while len(train_dataset)<TRAIN_SIZE:
        dp = next(gen)
        if dp!=None:
            train_dataset.append(dp)
            hits+=1
        else:
            misses+=1
        if len(train_dataset)%100==0:
            print(f"created {len(train_dataset)} points...")
    print(f"Finished one split with yield={hits/(hits+misses)}")

    val_dataset = []
    hits, misses=0,0
    while len(val_dataset)<VAL_SIZE:
        dp = next(gen)
        if dp!=None:
            val_dataset.append(dp)
            hits+=1
        else:
            misses+=1
    print(f"Finished one split with yield={hits/(hits+misses)}")

    test_dataset = []
    hits, misses=0,0
    while len(test_dataset)<TEST_SIZE:
        dp = next(gen)
        if dp!=None:
            test_dataset.append(dp)
            hits+=1
        else:
            misses+=1
    print(f"Finished one split with yield={hits/(hits+misses)}")

    write_task_dataset("steptask-NextSentenceGeneration",
                       train_dataset,
                       val_dataset,
                       test_dataset)


    print("")


    #######################################################

    gen = MaskedDocumentGeneration()

    train_dataset = []
    hits, misses=0,0
    while len(train_dataset)<TRAIN_SIZE:
        dp = next(gen)
        if dp!=None:
            train_dataset.append(dp)
            hits+=1
        else:
            misses+=1
        if len(train_dataset)%100==0:
            print(f"created {len(train_dataset)} points...")
    print(f"Finished one split with yield={hits/(hits+misses)}")

    val_dataset = []
    hits, misses=0,0
    while len(val_dataset)<VAL_SIZE:
        dp = next(gen)
        if dp!=None:
            val_dataset.append(dp)
            hits+=1
        else:
            misses+=1
    print(f"Finished one split with yield={hits/(hits+misses)}")

    test_dataset = []
    hits, misses=0,0
    while len(test_dataset)<TEST_SIZE:
        dp = next(gen)
        if dp!=None:
            test_dataset.append(dp)
            hits+=1
        else:
            misses+=1
    print(f"Finished one split with yield={hits/(hits+misses)}")

    write_task_dataset("steptask-MaskedDocumentGeneration",
                       train_dataset,
                       val_dataset,
                       test_dataset)


    print("")

    ###############################################################


    gen = SentenceReordering()

    train_dataset = []
    hits, misses=0,0
    while len(train_dataset)<TRAIN_SIZE:
        dp = next(gen)
        if dp!=None:
            train_dataset.append(dp)
            hits+=1
        else:
            misses+=1
        if len(train_dataset)%100==0:
            print(f"created {len(train_dataset)} points...")
    print(f"Finished one split with yield={hits/(hits+misses)}")

    val_dataset = []
    hits, misses=0,0
    while len(val_dataset)<VAL_SIZE:
        dp = next(gen)
        if dp!=None:
            val_dataset.append(dp)
            hits+=1
        else:
            misses+=1
    print(f"Finished one split with yield={hits/(hits+misses)}")

    test_dataset = []
    hits, misses=0,0
    while len(test_dataset)<TEST_SIZE:
        dp = next(gen)
        if dp!=None:
            test_dataset.append(dp)
            hits+=1
        else:
            misses+=1
    print(f"Finished one split with yield={hits/(hits+misses)}")

    write_task_dataset("steptask-SentenceReordering",
                       train_dataset,
                       val_dataset,
                       test_dataset)



