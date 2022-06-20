import json
from collections import Counter, defaultdict
from copy import deepcopy
from typing import List
from summarization_tasks_datagen.base import BaseGenerator, RealWorldGenerator
from overrides import overrides
import numpy as np
import jsonlines
import os
from tqdm import tqdm


class Modifier():
    def __init__(self):
        if hasattr(self, "keywords"):
            self.distinctify_keywords()

    def set_id(self, id):
        if not hasattr(self, "id"):
            self.id=id

    def distinctify_keywords(self):
        new_keywords = [f"__{self.id}__{k}__" for k in self.keywords]
        self.keywords = new_keywords

    def modify_input(self, dp):
        raise NotImplementedError

    def get_output(self, dp):
        raise NotImplementedError

    def get_random_sent_indices(self, article_lines, num_sent):
        replacement_line_indices = np.random.choice(range(len(article_lines)), (num_sent,), replace=False)
        replacement_line_indices = sorted(replacement_line_indices)
        return replacement_line_indices

    def find_line_indices_with_keywords(self, article_lines):
        out_line_indices = []
        detected_keywords = []
        for (i, line) in enumerate(article_lines):
            words = set(line.split(" "))
            common_words = set(self.keywords).intersection(words)
            if len(common_words)>0:
                out_line_indices.append(i)
                detected_keywords.append(list(common_words))

        return out_line_indices, detected_keywords


    def get_replacement_candidates(self, article_lines, num_sent):
        replacement_line_indices = self.get_random_sent_indices(article_lines, num_sent)
        replaceable_word_indices = []

        for ind in replacement_line_indices:
            line = article_lines[ind]
            words = line.split(" ")
            replaceable_indices = [i for (i,w) in enumerate(words[:-1]) if ("_" not in w)]
            replaceable_word_indices.append(replaceable_indices)

        return replacement_line_indices, replaceable_word_indices


# Copying first sentence
class CopyFirstSentence(Modifier):
    def __init__(self):
        self.set_id("d1")
        super().__init__()

    @overrides
    def modify_input(self, dp):
        pass

    @overrides
    def get_output(self, dp):
        return [dp["article_lines"][0]]


# Copying last sentence
class CopyLastSentence(Modifier):
    def __init__(self):
        self.set_id("d2")
        super().__init__()

    @overrides
    def modify_input(self, dp):
        pass

    @overrides
    def get_output(self, dp):
        return [dp["article_lines"][-1]]


# Copying all phrases/sentences that contain important keyword
class CopyKeywordOneSentence(Modifier):
    @overrides
    def __init__(self):
        self.set_id("d3")
        NUM_KEYWORDS = 10
        self.keywords = [f"keyword_{i}" for i in range(NUM_KEYWORDS)]
        super().__init__()

    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]
        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, 1)

        for (rep_line_index, allowed_rep_word_indices) in zip(replacement_line_indices, replaceable_word_indices):
            words = article_lines[rep_line_index].split(" ")
            rep_word_index = np.random.choice(allowed_rep_word_indices)
            words[rep_word_index] = np.random.choice(self.keywords)
            article_lines[rep_line_index] = " ".join(words)

    @overrides
    def get_output(self, dp):
        article_lines = dp["article_lines"]
        out_line_indices, detected_keywords = self.find_line_indices_with_keywords(article_lines)
        assert len(out_line_indices)==1
        out_lines = [article_lines[out_line_indices[0]]]
        return out_lines


# Copying all phrases/sentences that contain important keywords in original order
class CopyKeywordMultipleSentences(CopyKeywordOneSentence):
    @overrides
    def __init__(self):
        self.set_id("d4")
        super().__init__()

    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]

        num_to_replace = 3
        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, num_to_replace)

        for (rep_line_index, allowed_rep_word_indices) in zip(replacement_line_indices, replaceable_word_indices):
            words = article_lines[rep_line_index].split(" ")
            rep_word_index = np.random.choice(allowed_rep_word_indices)
            words[rep_word_index] = np.random.choice(self.keywords)
            article_lines[rep_line_index] = " ".join(words)

    @overrides
    def get_output(self, dp):
        article_lines = dp["article_lines"]
        out_line_indices, detected_keywords = self.find_line_indices_with_keywords(article_lines)
        out_line_indices = sorted(out_line_indices)
        out_lines = [article_lines[ind] for ind in out_line_indices]
        return out_lines


# Copying all phrases/sentences that contain important keywords in random order
class CopyKeywordMultipleSentencesShuffled(CopyKeywordMultipleSentences):
    @overrides
    def __init__(self):
        self.set_id("d5")
        super().__init__()

    @overrides
    def get_output(self, dp):
        out_lines = super().get_output(dp)
        np.random.shuffle(out_lines)
        return out_lines


# Ordering information in the right order or chronologically
class CopyKeywordMultipleSentencesSorted(CopyKeywordMultipleSentences):
    @overrides
    def __init__(self):
        self.set_id("d6")
        super().__init__()

    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]

        num_to_replace = 3
        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, num_to_replace)
        all_words_to_put = np.random.choice(self.keywords, (num_to_replace,), replace=False)

        for (rep_line_index, allowed_rep_word_indices, word_to_put) in zip(replacement_line_indices, replaceable_word_indices, all_words_to_put):
            words = article_lines[rep_line_index].split(" ")
            rep_word_index = np.random.choice(allowed_rep_word_indices)
            words[rep_word_index] = word_to_put
            article_lines[rep_line_index] = " ".join(words)

    @overrides
    def get_output(self, dp):
        article_lines = dp["article_lines"]
        out_line_indices, detected_keywords = self.find_line_indices_with_keywords(article_lines)

        order = np.argsort([x[0] for x in detected_keywords])

        out_lines = [article_lines[out_line_indices[ind]] for ind in order]
        return out_lines


# Copying sentence within quotes
class CopyQuoted(Modifier):
    @overrides
    def __init__(self):
        self.set_id("d7")
        self.keywords = ["begin_quote", "end_quote"]
        super().__init__()


    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]

        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, 1)

        for (rep_line_index, allowed_rep_word_indices) in zip(replacement_line_indices, replaceable_word_indices):
            words = article_lines[rep_line_index].split(" ")
            if 0 in allowed_rep_word_indices:
                begin_quote_word_index = np.random.randint(0, len(words))
            else:
                begin_quote_word_index = np.random.randint(1, len(words))
            end_quote_word_index = np.random.randint(begin_quote_word_index+1, len(words)+1)
            prev_words = words[:begin_quote_word_index]
            inner_words = words[begin_quote_word_index:end_quote_word_index]
            later_words = words[end_quote_word_index:]

            new_words = prev_words + [self.keywords[0]] + inner_words + [self.keywords[1]] + later_words
            article_lines[rep_line_index] = " ".join(new_words)



    @overrides
    def get_output(self, dp):
        article_lines = dp["article_lines"]

        line_indices, _ = self.find_line_indices_with_keywords(article_lines)
        assert len(line_indices)==1

        summary_lines = []
        for idx in line_indices:
            line = article_lines[idx]
            words = line.split(" ")
            open_quote_index = words.index(self.keywords[0])
            close_quote_index = words.index(self.keywords[1])
            inside_words = words[open_quote_index+1:close_quote_index]
            out_sent = " ".join(inside_words)
            if out_sent[-1]!=".":
                out_sent = out_sent + " ."
            summary_lines.append(out_sent)

        return summary_lines



# Copying content in bullet points
class CopyBulleted(Modifier):
    @overrides
    def __init__(self):
        self.set_id("d8")
        self.keywords = ["bullet"]
        super().__init__()


    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]
        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, 1)
        for (rep_line_index, allowed_rep_word_indices) in zip(replacement_line_indices, replaceable_word_indices):
            article_lines[rep_line_index] = self.keywords[0]+" "+article_lines[rep_line_index]


    @overrides
    def get_output(self, dp):
        article_lines = dp["article_lines"]

        line_indices, _ = self.find_line_indices_with_keywords(article_lines)
        assert len(line_indices)==1

        summary_lines = []
        for idx in line_indices:
            line = article_lines[idx]
            words = line.split(" ")
            assert words[0]==self.keywords[0]
            summary_lines.append(" ".join(words[1:]))

        return summary_lines



# Check if keyword or phrase occurs in the input
class CheckKeyword(Modifier):
    @overrides
    def __init__(self):
        self.set_id("d9")
        self.keywords = ["keyword"]
        super().__init__()


    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]
        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, 1)

        if np.random.randint(0,2)==1:
            line_to_replace = replacement_line_indices[0]
            index_to_replace = np.random.choice(replaceable_word_indices[0])
            line = article_lines[line_to_replace]
            words = line.split(" ")
            words[index_to_replace] = self.keywords[0]
            article_lines[line_to_replace] = " ".join(words)

    def get_output(self, dp):
        article_lines = dp["article_lines"]

        line_indices, _ = self.find_line_indices_with_keywords(article_lines)

        assert len(line_indices) in [0,1]

        if len(line_indices)==1:
            summary_lines = ["the special word was found ."]
        elif len(line_indices)==0:
            summary_lines =["the special word was not found ."]

        return summary_lines



# Classifying a class of adjectives into positive/negative (varies across products)
class ClassifyKeyword(Modifier):
    @overrides
    def __init__(self):
        self.set_id("d10")
        self.keywords = [f"keyword_{q}" for q in range(10)]
        super().__init__()
        self.pos_keywords = self.keywords[:5]
        self.neg_keywords = self.keywords[5:]


    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]
        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, 1)

        line_to_replace = replacement_line_indices[0]
        index_to_replace = np.random.choice(replaceable_word_indices[0])
        line = article_lines[line_to_replace]
        words = line.split(" ")
        words[index_to_replace] = np.random.choice(self.keywords)
        article_lines[line_to_replace] = " ".join(words)


    @overrides
    def get_output(self, dp):
        article_lines = dp["article_lines"]

        line_indices, keywords_found = self.find_line_indices_with_keywords(article_lines)
        assert len(line_indices)==1
        kw = keywords_found[0][0]

        summary_lines = []
        if kw in self.pos_keywords:
            summary_lines.append("the keyword was positive .")
        elif kw in self.neg_keywords:
            summary_lines.append("the keyword was negative .")
        else:
            raise ValueError
        return summary_lines


# Replacing a set of objects by its class
class ReplaceClassKeyword(Modifier):
    @overrides
    def __init__(self):
        self.set_id("d11")
        self.num_classes = 3
        self.num_words_per_class = 3
        self.per_class_keywords = [ [f"class_{ci}_kw_{ki}" for ki in range(self.num_words_per_class)] for ci in range(self.num_classes)]
        self.keywords = sum(self.per_class_keywords, [])
        super().__init__()

        self.per_class_keywords = []
        kws = deepcopy(self.keywords)
        for i in range(self.num_classes):
            temp = []
            for j in range(self.num_words_per_class):
                temp.append(kws[0])
                kws = kws[1:]
            self.per_class_keywords.append(temp)

        self.classes = [ f"__{self.id}__class_{ci}__" for ci in range(self.num_classes)]


    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]
        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, 1)

        line_to_replace = replacement_line_indices[0]
        index_to_replace = np.random.choice(replaceable_word_indices[0])
        line = article_lines[line_to_replace]
        words = line.split(" ")
        words[index_to_replace] = np.random.choice(self.keywords)
        article_lines[line_to_replace] = " ".join(words)

    @overrides
    def get_output(self, dp):
        article_lines = dp["article_lines"]
        line_indices, keywords_found = self.find_line_indices_with_keywords(article_lines)
        assert len(line_indices)==1
        li = line_indices[0]
        kw = keywords_found[0][0]

        summary_lines = []
        for i in range(self.num_classes):
            if kw in self.per_class_keywords[i]:
                new_line = article_lines[li].replace(kw, self.classes[i])
                summary_lines.append(new_line)

        assert len(summary_lines)>0
        return summary_lines



# Figuring out majority opinion
class MajorityKeyword(Modifier):
    def __init__(self):
        self.set_id("d12")
        self.keywords = ["keyword_1", "keyword_2"]
        super().__init__()


    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]

        num_sents_to_modify = np.random.randint(1,3)
        num_sents_to_modify = 2*num_sents_to_modify+1   # making sure it's odd

        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, num_sents_to_modify)

        for (line_to_replace, indices_can_replace) in zip(replacement_line_indices, replaceable_word_indices):
            kw_to_add = np.random.choice(self.keywords)
            line = article_lines[line_to_replace]
            words = line.split(" ")
            index_to_replace = np.random.choice(indices_can_replace)
            words[index_to_replace] = kw_to_add
            article_lines[line_to_replace] = " ".join(words)

    @overrides
    def get_output(self, dp):
        article_lines = dp["article_lines"]
        line_indices, keywords_found = self.find_line_indices_with_keywords(article_lines)

        all_keywords_found = sum(keywords_found, [])
        kw_counter = Counter(all_keywords_found)
        majority_kw, _ = kw_counter.most_common(1)[0]
        majority_kw_idx = self.keywords.index(majority_kw)

        output_sent = f"The keyword {majority_kw_idx+1} occurred more number of times ."
        return [output_sent]


# Segregating information to different topics (e.g. aspects of biography or review)
class TopicSegregation(ReplaceClassKeyword):
    @overrides
    def __init__(self):
        self.set_id("d13")
        super().__init__()


    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]

        num_sents_to_modify = np.random.randint(2,6)
        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, num_sents_to_modify)

        for (line_to_replace, word_indices_to_replace) in zip(replacement_line_indices, replaceable_word_indices):
            line = article_lines[line_to_replace]
            words = line.split(" ")
            index_to_replace = np.random.choice(word_indices_to_replace)
            words[index_to_replace] = np.random.choice(self.keywords)
            article_lines[line_to_replace] = " ".join(words)


    @overrides
    def get_output(self, dp):
        article_lines = dp["article_lines"]
        line_indices, keywords_found = self.find_line_indices_with_keywords(article_lines)

        kw_index_dict=defaultdict(list)

        for (li, kws) in zip(line_indices, keywords_found):
            kw = kws[0]
            for i in range(self.num_classes):
                if kw in self.per_class_keywords[i]:
                    kw_index_dict[i].append(li)

        output_lines = []
        for (class_index, class_name) in enumerate(self.classes):
            output_lines.append(f"section {class_name}")
            for rel_line_index in sorted(kw_index_dict[class_index]):
                output_lines.append(article_lines[rel_line_index])

        return output_lines


# Deciding if a number is over a threshold
class ThresholdNumber(Modifier):
    def __init__(self):
        self.set_id("d14")
        self.num_numbers = 100
        self.keywords = [str(x) for x in range(self.num_numbers)]
        super().__init__()

    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]
        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, 1)

        line_to_replace = replacement_line_indices[0]
        index_to_replace = np.random.choice(replaceable_word_indices[0])
        line = article_lines[line_to_replace]
        words = line.split(" ")
        words[index_to_replace] = np.random.choice(self.keywords)
        article_lines[line_to_replace] = " ".join(words)


    @overrides
    def get_output(self, dp):
        article_lines = dp["article_lines"]

        line_indices, keywords_found = self.find_line_indices_with_keywords(article_lines)
        assert len(line_indices)==1
        kw = keywords_found[0][0]

        summary_lines = []
        if kw in self.keywords[self.num_numbers//2:]:
            summary_lines.append("the number was above or equal to threshold .")
        else:
            summary_lines.append("the number was below threshold .")

        return summary_lines



# Finding which of two numbers is greater
class CompareNumbers(ThresholdNumber):
    def __init__(self):
        self.set_id("d15")
        super().__init__()

    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]
        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, 2)

        kws_to_add = np.random.choice(self.keywords, (2,), replace=False)
        for (line_to_replace, indices_can_replace, kw) in zip(replacement_line_indices, replaceable_word_indices, kws_to_add):
            index_to_replace = np.random.choice(indices_can_replace)
            line = article_lines[line_to_replace]
            words = line.split(" ")
            words[index_to_replace] = kw
            article_lines[line_to_replace] = " ".join(words)

    def get_output(self, dp):
        article_lines = dp["article_lines"]
        line_indices, keywords_found = self.find_line_indices_with_keywords(article_lines)
        assert len(keywords_found)==2
        assert len(keywords_found[0])==1
        assert len(keywords_found[1])==1

        first_num_kw = keywords_found[0][0]
        second_num_kw = keywords_found[1][0]

        first_num = self.keywords.index(first_num_kw)
        second_num = self.keywords.index(second_num_kw)

        if first_num > second_num:
            output_sent = f"The first number is greater ."
        else:
            output_sent = f"The second number is greater ."

        return [output_sent]


# Finding largest  number(s)
class LargestNumber(ThresholdNumber):
    def __init__(self):
        self.set_id("d16")
        super().__init__()

    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]
        num_sents_to_modify = np.random.randint(1,5)
        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, num_sents_to_modify)
        kws_to_add = np.random.choice(self.keywords, (num_sents_to_modify,), replace=False)
        for (line_to_replace, indices_can_replace, kw) in zip(replacement_line_indices, replaceable_word_indices, kws_to_add):
            index_to_replace = np.random.choice(indices_can_replace)
            line = article_lines[line_to_replace]
            words = line.split(" ")
            words[index_to_replace] = kw
            article_lines[line_to_replace] = " ".join(words)

    def get_output(self, dp):
        article_lines = dp["article_lines"]
        line_indices, keywords_found = self.find_line_indices_with_keywords(article_lines)

        numbers_found = []
        for kw_list in keywords_found:
            assert len(kw_list)==1
            num_str = kw_list[0]
            numbers_found.append(self.keywords.index(num_str))

        output_sent = f"The largest number is {max(numbers_found)} ."
        return [output_sent]


# Compute the sum of numerical observations which are divided into different categories
class SumOfNumbers(LargestNumber):
    def __init__(self):
        self.set_id("d17")
        super().__init__()

    def get_output(self, dp):
        article_lines = dp["article_lines"]
        line_indices, keywords_found = self.find_line_indices_with_keywords(article_lines)

        numbers_found = []
        for kw_list in keywords_found:
            assert len(kw_list)==1
            num_str = kw_list[0]
            numbers_found.append(self.keywords.index(num_str))

        output_sent = f"The sum of numbers is {sum(numbers_found)} ."
        return [output_sent]

# Using paraphrases of words
class ParaphraseWords(Modifier):
    def __init__(self):
        self.set_id("d18")

        self.num_src_words = 20
        self.num_variants_per_word = 3
        self.keywords = []

        for src_index in range(self.num_src_words):
            src_word = f"src_{src_index}"
            self.keywords.append(src_word)
            for targ_index in range(self.num_variants_per_word):
                targ_word = f"target_{src_index}_{targ_index}"
                self.keywords.append(targ_word)
        super().__init__()

        self.src_words = []
        self.paraphrase_maps = {}
        last_src_kw = None
        for kw in self.keywords:
            if "src" in kw:
                self.src_words.append(kw)
                self.paraphrase_maps[kw]=[]
                last_src_kw = kw
            else:
                self.paraphrase_maps[last_src_kw].append(kw)

    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]
        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, 1)
        line_to_replace = replacement_line_indices[0]
        index_to_replace = np.random.choice(replaceable_word_indices[0])
        line = article_lines[line_to_replace]
        words = line.split(" ")
        words[index_to_replace] = np.random.choice(self.src_words)
        article_lines[line_to_replace] = " ".join(words)


    @overrides
    def get_output(self, dp):
        article_lines = dp["article_lines"]
        line_indices, keywords_found = self.find_line_indices_with_keywords(article_lines)

        summary_lines = []
        for (line_index, kw_list) in zip(line_indices, keywords_found):
            assert len(kw_list)==1
            kw = kw_list[0]
            assert kw in self.src_words
            line = article_lines[line_index]
            words = line.split(" ")
            word_to_replace_index = words.index(kw)
            possible_targets = self.paraphrase_maps[kw]
            target_word = np.random.choice(possible_targets)
            words[word_to_replace_index] = target_word
            summary_lines.append(" ".join(words))

        return summary_lines


# Joining full-clauses from different sentences
class JoinClauses(Modifier):
    @overrides
    def __init__(self):
        self.set_id("d19")
        self.num_joiners = 5
        self.keywords = [f"join_{idx}" for idx in range(self.num_joiners)]
        super().__init__()


    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]

        num_to_replace = 3
        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, num_to_replace)
        joiners = np.random.choice(self.keywords, (num_to_replace,), replace=False)


        for (replacing_line_index, replacing_word_indices, joiner) in \
                zip(replacement_line_indices, replaceable_word_indices, joiners):
            replacement_line = article_lines[replacing_line_index]
            words = replacement_line.split(" ")
            allowed_replacement_indices = [x for x in replacing_word_indices if x<len(words)-2]
            assert len(allowed_replacement_indices)>0
            replacement_word_index = np.random.choice(allowed_replacement_indices)
            words[replacement_word_index] = joiner
            article_lines[replacing_line_index] = " ".join(words)

    @overrides
    def get_output(self, dp):
        article_lines = dp["article_lines"]
        line_indices, keywords_found = self.find_line_indices_with_keywords(article_lines)

        output_sent = "The combined sentence is "
        for (line_index, kw_list) in zip(line_indices, keywords_found):
            assert len(kw_list)==1
            kw = kw_list[0]
            line = article_lines[line_index]
            words = line.split(" ")
            kw_index = words.index(kw)
            following_words = words[kw_index+1:]
            following_words = following_words[:-1]
            output_sent += kw + " " + " ".join(following_words) + " "

        output_sent += "."
        return [output_sent]

# Aggregating information of event - what, when, where (subsumed?)


# Breaking down conjugated sentence into different clauses
class BreakClauses(Modifier):
    @overrides
    def __init__(self):
        self.set_id("d20")
        self.num_joiners = 5
        self.keywords = [f"join_{idx}" for idx in range(self.num_joiners)]
        super().__init__()

    @overrides
    def modify_input(self, dp):
        article_lines = dp["article_lines"]

        num_to_replace = 1
        replacement_line_indices, replaceable_word_indices = self.get_replacement_candidates(article_lines, num_to_replace)
        joiners = np.random.choice(self.keywords, (2,), replace=False)

        replacement_line_index = replacement_line_indices[0]
        replacing_word_indices = replaceable_word_indices[0]

        line = article_lines[replacement_line_index]
        words = line.split(" ")
        allowed_replacement_indices = [x for x in replacing_word_indices if x<len(words)-1]
        assert len(allowed_replacement_indices)>=2
        replacement_word_indices = np.random.choice(allowed_replacement_indices, (2,), replace=False)

        words[replacement_word_indices[0]] = joiners[0]
        words[replacement_word_indices[1]] = joiners[1]
        article_lines[replacement_line_index] = " ".join(words)

    @overrides
    def get_output(self, dp):
        article_lines = dp["article_lines"]
        line_indices, keywords_found = self.find_line_indices_with_keywords(article_lines)

        assert len(line_indices)==1
        kws = keywords_found[0]
        assert len(kws) == 2

        line_index = line_indices[0]
        line = article_lines[line_index]
        words = line.split(" ")
        first_kw_index = words.index(kws[0])
        second_kw_index = words.index(kws[1])

        if first_kw_index>second_kw_index:  # reverse the orders of the keywords if necessary
            kws = [kws[1], kws[0]]
            insert_poses = [second_kw_index, first_kw_index]
        else:
            kws = [kws[0], kws[1]]
            insert_poses = [first_kw_index, second_kw_index]

        summary_lines = []
        prefix = words[:insert_poses[0]]
        for _i in range(len(insert_poses)):
            start = insert_poses[_i]
            if _i==len(insert_poses)-1:
                end = len(words) - 1
            else:
                end = insert_poses[_i+1]
            summary_lines.append(" ".join(prefix+words[start:end]+["."]))

        return summary_lines



# Truncating sentence by dropping phrase at the end or in the middle
class TruncateSentence(CopyKeywordOneSentence):
    @overrides
    def __init__(self):
        self.set_id("d21")
        super().__init__()

    @overrides
    def get_output(self, dp):
        article_lines = dp["article_lines"]
        out_line_indices, detected_keywords = self.find_line_indices_with_keywords(article_lines)

        assert len(out_line_indices)==1
        kw = detected_keywords[0][0]
        line_index = out_line_indices[0]
        line = article_lines[line_index]
        words = line.split(" ")
        keyword_index = words.index(kw)
        output_words = words[:keyword_index] + ["."]

        out_lines = [" ".join(output_words)]
        return out_lines


def write_task_dataset(task_name, trainset, valset, testset, task_names):
    if not os.path.exists("../dataset_root/pretraining_datasets"):
        os.mkdir("../dataset_root/pretraining_datasets")
    base_path = f"../dataset_root/pretraining_datasets/{task_name}"

    os.mkdir(base_path)
    json.dump(task_names, open(f"{base_path}/task_names.json","w"))
    for (splitname, split) in zip(["train","val","test"], [trainset, valset, testset]):
        with jsonlines.open(os.path.join(base_path, f"{splitname}.jsonl"), "w") as w:
            for dp in split:
                w.write(dp)


GENERATOR = BaseGenerator()
WAS_SET_VOCAB_CALLED = False           
def gen_one_summ_data(mode, length_range, char_set=None, vocab_size=None, num_chars_per_word=None, mean_numsents=10, mean_sentlen=10):
    # print(GENERATOR.tokens)
    task = eval(mode + '()')        
    global WAS_SET_VOCAB_CALLED
    if char_set is not None and not WAS_SET_VOCAB_CALLED:
        GENERATOR.set_vocab(char_set, vocab_size, num_chars_per_word)
        WAS_SET_VOCAB_CALLED = True
    data = {"article_lines":GENERATOR.get_para(mean_numsents, mean_sentlen),
                "summary_lines":[]}
    def split(word):
        return [char for char in word]
    
    input = data['article_lines']
    task.modify_input(data)
    input = [','.join(sentence.split(' ')) for sentence in input]
    input = [' '.join(split(sentence)) for sentence in input]
    input = ' '.join(input)
    input_len = len(input.replace(' ', ''))
    while input_len < length_range[0] or input_len > length_range[1]:
        data = {"article_lines":GENERATOR.get_para(mean_numsents, mean_sentlen),
                "summary_lines":[]}
        task.modify_input(data)
        input = data['article_lines']
        input = [','.join(sentence.split(' ')) for sentence in input]
        input = [' '.join(split(sentence)) for sentence in input]
        input = ' '.join(input)
        input_len = len(input.replace(' ', ''))

    output = task.get_output(data)
    output = [','.join(sentence.split(' ')) for sentence in output]
    output = [' '.join(split(sentence)) for sentence in output]
    output = ' '.join(output)

    return input, output


if __name__=="__main__":
    TRAIN_SIZE = 100000
    VAL_SIZE = 5000
    TEST_SIZE = 5000
    TOTAL_DATASET_SIZE = TRAIN_SIZE+VAL_SIZE+TEST_SIZE

    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #<<<<<<<<<<<<<<<<<<<<<<< Use this part for using nonsense documents to create pretraining datapoints
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    composite_task_name = "ourtasks-nonsense"
    mean_numsents:int = 10
    mean_sentlen:int = 10
    generator = BaseGenerator()
    dataset = [{"article_lines":generator.get_para(mean_numsents, mean_sentlen),
                "summary_lines":[]}
                                            for _ in tqdm(range(TOTAL_DATASET_SIZE))]
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #<<<<<<<<<<<<<<<<<<<<<< Use this part for using wikipedia documents to create pretraining datapoints
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # composite_task_name = "ourtasks-wiki"
    # generator = RealWorldGenerator(dataset_name="wikipedia")
    # dataset = []
    # hits=0
    # misses=0
    # print(TOTAL_DATASET_SIZE)
    # while len(dataset)<TOTAL_DATASET_SIZE:
    #     article_lines = generator.get_para()
    #     if article_lines==None:
    #         misses+=1
    #         continue
    #     else:
    #         dataset.append({"article_lines":article_lines,
    #                         "summary_lines":[]})
    #         hits+=1
    #         if len(dataset)%1000==0:
    #             print(f"{len(dataset)} points done. Wastage = {misses/(hits+misses)}")
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # classes for tasks to be used
    _classes = [
                # CopyFirstSentence,  # does not modify input
                # CopyLastSentence, # does not modify input
                CopyKeywordOneSentence,
                CopyKeywordMultipleSentences,
                CopyKeywordMultipleSentencesShuffled,
                CopyKeywordMultipleSentencesSorted,
                CopyQuoted,
                CopyBulleted,
                # CheckKeyword, # does not modify input
                ClassifyKeyword,
                ReplaceClassKeyword,
                MajorityKeyword,
                TopicSegregation,
                ThresholdNumber,
                # CompareNumbers, # could not be learnt even in isolation by t5
                LargestNumber,
                # SumOfNumbers, # could not be learnt even in isolation by t5
                ParaphraseWords,
                JoinClauses,
                BreakClauses,
                TruncateSentence
               ]


    NUM_TRANSFORMS = 3
    task_names = [c.__name__ for c in _classes]
    print("trying out tasks = ", task_names)

    _classes = [x() for x in _classes]  # instantitate objects
    new_dataset=[]

    for dp in tqdm(dataset):
        classes_to_use = np.random.choice(_classes, size=(NUM_TRANSFORMS,), replace=False)
        for _class in classes_to_use:
            _class.modify_input(dp)
        for _class in classes_to_use:
            output = _class.get_output(dp)
            dp["summary_lines"].extend(output)


    write_task_dataset(composite_task_name,
                       dataset[:TRAIN_SIZE],
                       dataset[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE],
                       dataset[-TEST_SIZE:],
                       task_names)


