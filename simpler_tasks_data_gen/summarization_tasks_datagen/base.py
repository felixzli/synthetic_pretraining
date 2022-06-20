from summarization_tasks_datagen.vocab import CustomVocab, RandomVocab, PretrainedT5Vocab
# from transformers import T5Tokenizer
import numpy as np
import nltk
# from datasets import load_dataset

class BaseGenerator():
    def __init__(self, numsent_tolerance = 3, sentlen_tolerance = 5):
        '''vocab_type is either RandomVocab or PretrainedT5Vocab'''
        self.numsent_tolerance = numsent_tolerance
        self.sentlen_tolerance = sentlen_tolerance

        self.vocab = RandomVocab()
        tokens = self.vocab.tokens
        self.tokens = [tok for tok in tokens if "." not in tok]


    def set_vocab(self, chars, vocab_size, chars_per_word):
        vocab = CustomVocab(list(chars), vocab_size, chars_per_word)
        self.vocab = vocab
        tokens = self.vocab.tokens
        self.tokens = [tok for tok in tokens if "." not in tok]


    def gen_sent(self, num_toks):
        sampled_tokens = np.random.choice(self.tokens, (num_toks,), replace=True)
        if type(self.vocab)==RandomVocab or type(self.vocab)==CustomVocab:
            return " ".join(sampled_tokens)
        elif type(self.vocab)==PretrainedT5Vocab:
            return self.tokenizer.convert_tokens_to_string(sampled_tokens.tolist())
        else:
            raise NotImplementedError


    def get_para(self, mean_numsents, mean_sentlen):
        numsents = mean_numsents + np.random.randint(-self.numsent_tolerance, self.numsent_tolerance+1)
        numsents = max(1, numsents)

        sentences = []
        for sent_index in range(numsents):
            sentlen = mean_sentlen + np.random.randint(-self.sentlen_tolerance, self.sentlen_tolerance+1)
            sentlen = max(1, sentlen)
            sent = self.gen_sent(sentlen)
            sentences.append(sent+" .")

        return sentences


class RealWorldGenerator(object):
    def __init__(self, dataset_name="wikipedia", mean_numsents=10, numsent_tolerance=3):
        self.dataset_name = dataset_name
        if self.dataset_name=="wikipedia":
          self.dataset = load_dataset('wikipedia', "20200501.en",  split=f'train')
        elif self.dataset_name=="bookcorpus":
          self.dataset = load_dataset('bookcorpus', split=f'train')
        else:
          raise NotImplementedError
        self.numsent_tolerance = numsent_tolerance
        self.dataset_iterator = self.next_doc_iterator()
        self.mean_numsents = mean_numsents

    def next_doc_iterator(self):
        for dp in self.dataset:
            yield dp

    def get_para(self):
        dp = next(self.dataset_iterator)
        txt = dp["text"]
        numsents = self.mean_numsents + np.random.randint(-self.numsent_tolerance, self.numsent_tolerance+1)
        numsents = max(1, numsents)
        sents = nltk.sent_tokenize(txt)
        sents = sents[:numsents]
        if len(sents)!=numsents:
            return None           # too  few sentences in the document

        tokenized_sents = []
        for sent in sents:
            words = nltk.word_tokenize(sent.strip())
            words = [w.replace(" ","") for w in words]  # removing spaces inside words if they occur by chance
            tokenized_sents.append(words)

        total_num_words = sum(len(x) for x in tokenized_sents)
        MAX_NUM_WORDS = (self.mean_numsents+self.numsent_tolerance)*(10+5)  # standard sentlen of 10 + tolerance of 5 as used in other experiments with BaseGenerator
        MIN_NUM_WORDS = (self.mean_numsents-self.numsent_tolerance)*(10-5)  # standard sentlen of 10 + tolerance of 5 as used in other experiments with BaseGenerator
        if total_num_words>MAX_NUM_WORDS or total_num_words<MIN_NUM_WORDS:
            return None
        min_words_in_any_sent = min(len(x) for x in tokenized_sents)
        if min_words_in_any_sent<5:
            return None

        sentences = [" ".join(x).lower() for x in tokenized_sents]

        return sentences



class RealWorldGeneratorForBaseline(object):
    def __init__(self, dataset_name="wikipedia", required_tokens=1024):
        self.dataset_name = dataset_name
        if self.dataset_name=="wikipedia":
          ds = load_dataset('wikipedia', "20200501.en",  split=f'train')
          self.dataset = ds.shuffle(seed=np.random.randint(32768))
        elif self.dataset_name=="bookcorpus":
          ds = load_dataset('bookcorpus', split=f'train')
          self.dataset = ds.shuffle(seed=np.random.randint(32768))
        else:
          raise NotImplementedError
        self.dataset_iterator = self.next_doc_iterator()
        self.required_tokens = required_tokens
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")

    def next_doc_iterator(self):
        for dp in self.dataset:
            yield dp

    def get_para(self):
        dp = next(self.dataset_iterator)
        txt = dp["text"].lower()

        sents = []
        num_toks=0
        valid=False
        for sent in nltk.sent_tokenize(txt):
            tokens = self.tokenizer.tokenize(sent)
            num_toks+=len(tokens)
            sents.append(sent)
            if num_toks>self.required_tokens:
                valid=True
                break

        if not valid or (num_toks>self.required_tokens+50): # second condition makes sure the last sentence is not too long
            # print(num_toks)
            return None

        return sents




if __name__=="__main__":
    gen = RealWorldGeneratorForBaseline(required_tokens=768)
    hits=0
    misses=0
    valid_points = []
    while len(valid_points) < 1000:
        article_lines = gen.get_para()
        if article_lines != None:
            hits += 1
            valid_points.append(article_lines)
            if len(valid_points)%100 == 0:
                print(f"{len(valid_points)} done")
        else:
            misses += 1

    print("wastage percentage = ",misses/(hits+misses))

