import nltk

# nltk.download('wordnet')
import spacy

# spacy.cli.download("en_core_web_lg")
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import sys
import os
import time
from pathlib import Path

import pandas as pd
import numpy as np
from collections import defaultdict


nlp = spacy.load("en_core_web_lg")
model_st = SentenceTransformer("bert-large-nli-mean-tokens")


class Boost_Prob(object):
    def __init__(self, topk_consider, penalty):
        self.topk_consider = topk_consider
        self.penalty = penalty
        self.sim_dict = defaultdict(dict)
        self.sim_one_round = defaultdict(int)
        self.stopwords = set()
        with open("data_story/stopwords.txt") as f:
            for word in f.readlines():
                self.stopwords.add(word.replace("\n", ""))

    def next_round_clean(self):
        self.sim_one_round = dict()

    def clear_cache(self):
        self.sim_dict = defaultdict(dict)
        self.sim_one_round = dict()

    def get_catolog_data(
        self, site_id, date="current", data_dir="/data/CQMlUtils/data"
    ):
        catalogs = c.Catalogs(site_id, data_dir=data_dir)
        catalog_date_df = catalogs.get_date(date)

    def generate_syn(self, word_lst, unseen_words, sim=0.9):
        res_ant = dict()
        res_syn = dict()
        for words in word_lst:
            for word in words.split():
                synonyms = [word]
                antonyms = []
                for syn in wordnet.synsets(word):
                    for l in syn.lemmas():
                        synonyms.append(l.name())
                        if l.antonyms():
                            antonyms.append(l.antonyms()[0].name())
                synonyms = [s for s in synonyms if type(s) == str and len(s) > 1] + [
                    w.lower() for w in word_lst
                ]
                antonyms = [s for s in antonyms if type(s) == str and len(s) > 1]
                res_syn[word] = set(synonyms) if synonyms else set()
                res_ant[word] = set(antonyms) if antonyms else set()
                # print(word, '=> synonyms => \n', set(synonyms), '\n')
                # print(word, '=> antonyms => \n',set(antonyms), '\n')
                # print('=' * 20)

        # only consider feature itself.
        res_syn = dict()
        for word in word_lst:
            res_syn[word] = set([word])
            if unseen_words:
                res_ant[word] = set(unseen_words)
        # print('res_syn', res_syn)
        # print('res_ant', res_ant)
        return res_syn, res_ant

    def sentence_transformers_sim(self, sentences1, sentences2):
        # Compute embedding for both lists

        embeddings1 = model_st.encode(
            sentences1, convert_to_tensor=True, show_progress_bar=False
        )
        embeddings2 = model_st.encode(
            sentences2, convert_to_tensor=True, show_progress_bar=False
        )

        # Compute cosine-similarits
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

        # Output the pairs with their score

        return [cosine_scores[i][i] for i in range(len(cosine_scores))]

    def define_sim(
        self,
        features,
        unseen_words,
        word,
        tokenizer,
        previous_id,
        sim=0.7,
        sentence_trans=False,
        no_repeat=True
    ):
        """
        features: list of strings: ['mesh', 'long sleeeve']; extracted from the image or ground truth
        word: str; the word we need to consider its similarity with features
        sim: int; when using sentence transformer, larger is harder to define sim
        sentence_trans: bool; whether to use sentence transformer. If not, we see whether word in features.
        """
        # print('word is %s and id is %s' % (word, str(word_id)), tokenizer.decode(word_id))
        # if previous_id != 0 and previous_id != 13 and previous_id != 50256:
        #     word_id = tokenizer.encode(' ' + word)
        # else:
        word_id = tokenizer.encode(word)
        # print(word, word_id, tokenizer.decode(previous_id))

        if not word:
            return 0

        if word in self.sim_one_round and no_repeat:
            return 0

        res_syn, res_ant = self.generate_syn(features, unseen_words)
        # print(res_syn, res_ant

        if sentence_trans:
            # syns
            syns_sig = False
            syns = " ".join([" ".join(list(res_syn[syn])) for syn in res_syn])

            tokens_feature = nlp(syns)
            tokens_words = nlp(word)
            for token_f in tokens_feature:
                for token_w in tokens_words:
                    if (
                        token_f.text in self.sim_dict[token_w.text]
                    ):  # consider the history
                        if self.sim_dict[token_w.text][token_f.text] > sim:
                            syns_sig = True
                            break

                    else:
                        similarity = self.sentence_transformers_sim(
                            [str(token_f)], [str(token_w)]
                        )[0]
                        self.sim_dict[token_w.text][token_f.text] = similarity
                        if token_f.text != token_w.text and similarity > sim:
                            print(
                                1,
                                token_f,
                                token_w,
                                similarity,
                            )
                            syns_sig = True
                            break
                if syns_sig:
                    break

            # ants
            ants_sig = False
            ants = " ".join([" ".join(list(res_ant[ant])) for ant in res_ant])
            tokens_feature = nlp(ants)
            for token_f in tokens_feature:
                for token_w in tokens_words:
                    if (
                        token_f.text in self.sim_dict[token_w.text]
                    ):  # consider the history
                        if self.sim_dict[token_w.text][token_f.text] > sim:
                            ants_sig = True
                            break
                    else:
                        similarity = self.sentence_transformers_sim(
                            [str(token_f)], [str(token_w)]
                        )[0]
                        self.sim_dict[token_w.text][token_f.text] = similarity
                        if token_f.text != token_w.text and similarity > sim:
                            print(
                                -1,
                                token_f,
                                token_w,
                                similarity,
                            )
                            ants_sig = True
                            break
                if ants_sig:
                    break
        else:
            # syns
            syns_sig = False
            syns = [" ".join(list(res_syn[syn])) for syn in res_syn]
            syns_ids = [tokenizer.encode(s) for s in syns]

            if word_id in syns_ids:
                # print("word %s in syns" % word)
                syns_sig = True
            else:
                for long_word in syns_ids:
                    # print('long_word', long_word)
                    if len(long_word) > 1:
                        # print('word_id[0]', word_id, long_word[0])
                        if word_id[0] == long_word[0]:
                            syns_sig = True
                        elif word_id[0] in long_word:
                            idx = long_word.index(word_id[0])
                            previous_word = tokenizer.decode(previous_id)
                            # print(long_word[idx - 1], word_id[0], tokenizer.encode(previous_word.strip()), tokenizer.encode(' ' + previous_word.strip()))
                            # print('previous_word.strip()', previous_word.strip(), tokenizer.encode(previous_word.strip()))
                            if (previous_word.strip()) and (long_word[idx - 1]
                                == tokenizer.encode(previous_word.strip())[0]
                                or long_word[idx - 1]
                                == tokenizer.encode(" " + previous_word.strip())[0]
                            ):
                                # print('idx', idx, long_word, word_id[0], tokenizer.encode(previous_word.strip()), tokenizer.encode(' ' + previous_word.strip()))
                                syns_sig = True
                # for feature in syns:
                #     if word.lower() in feature and word.strip().lower() not in self.stopwords:
                #         # print('syn => word %s in feature %s' % (word, feature))
                #         syns_sig = True
            # ants
            ants_sig = False
            ants = [" ".join(list(res_ant[ant])) for ant in res_ant]
            ants_ids = [tokenizer.encode(s) for s in ants]
            if word_id in ants_ids:
                # print('word %s in ants' % word)
                ants_sig = True
            else:
                for long_word in ants_ids:
                    if len(long_word) > 1:
                        if word_id[0] == long_word[0]:
                            ants_sig = True
                        elif word_id[0] in long_word:
                            idx = long_word.index(word_id[0])
                            if long_word[idx - 1] == previous_id:
                                ants_sig = True
                # for feature in ants:
                #     if word.lower() in feature and word.lower() not in self.stopwords:
                #         # print('ant => word %s in feature %s' % (word, feature))
                #         ants_sig = True

        if (ants_sig and syns_sig) or (not ants_sig and not syns_sig):
            self.sim_one_round[word] = 0
            return 0
        elif ants_sig and not syns_sig:
            self.sim_one_round[word] = -1
            return -1
        elif not ants_sig and syns_sig:
            self.sim_one_round[word] = 1
            return 1
        else:
            return

    def boost(
        self, tokenizer, probs, features, unseen_words=None, sim=0.9, previous_id=0
    ):
        # print('previous_id', previous_id, tokenizer.decode(previous_id), tokenizer.encode(' ' + tokenizer.decode(previous_id)))

        topk_lst = []
        ind = np.argpartition(probs, -self.topk_consider)[-self.topk_consider :]
        for word_token in ind.tolist():
            # print('word_token', word_token)
            topk_lst.append(tokenizer.decode([word_token]))

        boost_score = []
        probs = list(probs)
        done = 0
        for i, word in enumerate(topk_lst):
            if word.strip():
                no_space = self.define_sim(
                    features, unseen_words, word.strip(), tokenizer, previous_id, sim
                )
                space = self.define_sim(
                    features,
                    unseen_words,
                    " " + word.strip(),
                    tokenizer,
                    previous_id,
                    sim,
                )
                if no_space != 0 or space != 0:
                    done = 1
                    if no_space != 0:
                        boost_score.append(no_space)
                    else:
                        boost_score.append(space)
                else:
                    boost_score.append(0)
                # boost_score.append(self.define_sim(features, unseen_words, word.strip(), tokenizer, previous_id, sim))
                # if boost_score[-1] != 0:
                #     print('word is %s' % word, boost_score[-1], previous_id, tokenizer.decode(previous_id))
            else:
                boost_score.append(0)
        # boost prob
        # print(len(topk_lst))
        # print(len(ind.tolist()), ind)
        for i, idx in enumerate(ind):
            # if boost_score[i] != 0:
            # if boost_score[i] != 0:
            #     print('word is => ', topk_lst[i], ind[i])
            #     print('Boost score is =>', boost_score[i])
            #     print('Previous ids =>', previous_id, tokenizer.decode(previous_id))
            #     print('Probs before boosting is =>', probs[idx])
            probs[idx] = probs[idx] * (1 + boost_score[i] * self.penalty)
            # if boost_score[i] != 0:
            #     print('Probs after boosting is =>', probs[idx])
            #     print('-' * 20)
        # print(type(probs))
        return np.array(probs), done


if __name__ == "__main__":
    word_lst = [
        "Black",
        "Blouses",
        "Lace",
        "Long Sleeved",
        "Mesh",
        "Round Neck",
        "Sheer Tops",
        "Chiffon",
        "Leopard And Cheetah",
        "Spaghetti Straps",
    ]
    b = Boost_Prob(topk_consider=20, penalty=0.1)
    print(b.define_sim(features=word_lst, word="lion", sim=0.7, unseen_words="like"))
