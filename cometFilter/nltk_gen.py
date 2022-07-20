import spacy
from pattern.en import conjugate, SG, PL, INFINITIVE, PRESENT, PAST, FUTURE, pluralize, singularize, comparative, superlative, referenced
import argparse
import re
import itertools


class nltk_gen(object):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def read_tags(self, words):
        """
        print tags
        :param words:
        :return:
        """
        tokens = self.nlp(words)
        for token in tokens:
            print(token, token.tag_)

    def tense_detect(self, prompt):
        """
        Check the tense of verb
        """
        tokens = self.nlp(prompt)
        for token in tokens:
            if "VBD" == token.tag_:
                return "past"
        return "present"

    def verb_tense_gen(self, verb, tense, person, number):
        """

        :param verb: does not care whether it is lemma.
        :param tense: 'present', "past"
        :param person: 1 or 2 or 3
        :param number: "plural" or "singular"
        :return:
        """
        if len(verb.split(" ")) == 1:
            return conjugate(
                verb=str(verb),
                tense=tense,  # INFINITIVE, PRESENT, PAST, FUTURE
                person=int(person),  # 1, 2, 3 or None
                number=number,
            )  # SG, PL
        else:
            tokens = self.nlp(verb)
            res = ""
            for token in tokens:
                res = res + " " if res != "" else ""
                if "V" in token.tag_:
                    try:
                        res += conjugate(
                            verb=token.lemma_,
                            tense=tense,  # INFINITIVE, PRESENT, PAST, FUTURE
                            person=int(person),  # 1, 2, 3 or None
                            number=number,
                        )  # SG, PL
                    except StopIteration:
                        res += token.lemma_
                else:
                    res += token.text
            return res

    def nounFind(self, prompt):
        tokens = self.nlp(prompt)
        res = []
        for token in tokens:
            if "NN" in token.tag_:
                res.append(token.lemma_)
        return res if res else [prompt]

    def nounFindNone(self, prompt):
        tokens = self.nlp(prompt)
        res = []
        for token in tokens:
            if "NN" in token.tag_ and token.lemma_ not in ["personx", "none"]:
                res.append(token.lemma_)
        return res

    def find_verb_lemma(self, prompt):
        tokens = self.nlp(prompt)
        for token in tokens:
            if "V" in token.tag_:
                return token.lemma_

    def find_all_verb_lemma(self, prompt):
        tokens = self.nlp(prompt)
        res= []
        for token in tokens:
            if "V" in token.tag_:
                res.append(token.lemma_)
        return res

    def find_verb_phrase(self, prompt):
        if prompt.split(" ")[0] == "to":
            prompt = " ".join(prompt.split(" ")[1:])
        return prompt

    def noun_checker(self, phrase=None):
        tokens = self.nlp(phrase)
        for token in tokens:
            if "NN" in token.tag_:
                return True
        return False

    def verb_checker(self, phrase=None):
        tokens = self.nlp(phrase)
        for token in tokens:
            if "V" in token.tag_:
                return True
        return False

    def find_noun_lemma(self, words):
        tokens = self.nlp(words)
        for token in tokens:
            if "NN" in token.tag_:
                return token.lemma_
        return None

    def word_lemma(self, words):
        tokens = self.nlp(words)
        res = []
        for token in tokens:
            res.append(token.lemma_)
        return ' '.join(res) if res else None

    def noun_phrase_finder(self, words, sentence):
        """
        Given a sentence, return phrases
        :param sentence: a hammer and a saw
        :return: [hammer, saw]
        """
        tokens = self.nlp(sentence)
        res = []
        res_single = ''
        for i, token in enumerate(tokens):
            # print(token, token.tag_)
            if token.text in words or token.lemma_ in words:
                if "NN" in token.tag_:
                    res_single += token.lemma_ + ' '
                else:
                    if res_single:
                        res.append(res_single.strip())
                        res_single = ''
                    else:
                        pass
                if res_single:
                    res.append(res_single.strip())
        return res if res else []

    def noun_phrase_finder(self, words, sentence):
        """
        Given a sentence, return phrases
        :param sentence: a hammer and a saw
        :return: [hammer, saw]
        """
        tokens = self.nlp(sentence)
        res = []
        res_single = ''
        for i, token in enumerate(tokens):
            # print(token, token.tag_)
            if token.text in words or token.lemma_ in words:
                if "NN" in token.tag_:
                    res_single += token.lemma_ + ' '
                else:
                    if res_single:
                        res.append(res_single.strip())
                        res_single = ''
                    else:
                        pass
                if res_single:
                    res.append(res_single.strip())
        return res if res else []

    def adj_finder(self, words, sentence):
        """
        find adj and adv in the words and also in the sentence
        :param words:
        :param sentence:
        :return:
        """
        tokens = self.nlp(sentence)
        res = []
        res_single = ''
        for i, token in enumerate(tokens):
            # print(token, token.tag_)
            if token.text in words or token.lemma_ in words:
                if "RB" in token.tag_ or "JJ" in token.tag_:
                    res_single += token.lemma_ + ' '
                else:
                    if res_single:
                        res.append(res_single.strip())
                        res_single = ''
                    else:
                        pass
                if res_single:
                    res.append(res_single.strip())
        return res if res else []

    def verbs_all_tense(self, verb):
        res = set()
        for tense in [INFINITIVE, PRESENT, PAST, FUTURE]:
            for person in [1,2,3]:
                for number in [SG, PL]:
                    v = conjugate(
                        verb=str(verb),
                        tense=tense,  # INFINITIVE, PRESENT, PAST, FUTURE
                        person=int(person),  # 1, 2, 3 or None
                        number=number,
                    )
                    if v:
                        res.add(v)  # SG, PL
        # print(res)
        return res

    def noun_all_tense(self, noun):
        res = set()
        noun_p = pluralize(noun)
        if noun_p:
            res.add(noun_p)
        noun_s = singularize(noun)
        if noun_s:
            res.add(noun_s)
            res.add(referenced(noun_s))
        # print(res)
        return res

    def comet_to_delete(self, words, sentence):
        words = words.strip()
        if 'to ' == words[:3]:
            words = words[3:]
        # words = words.replace('personx', '')
        # words = words.replace('persony', '')
        words = words.strip()
        tense = self.tense_detect(sentence)
        tokens = self.nlp(words)
        res = ''
        for token in tokens:
            if 'V' in token.tag_:
                verb = self.verb_tense_gen(token.lemma_, tense, 3, 'singular')
                if verb == 'is':
                    pass
                else:
                    res += verb + ' '
            else:
                if 'PRP' in token.tag_:  #or 'NNP' in token.tag_:
                    pass
                else:
                    res += token.text + ' '
        return res.strip()

    def comet_to_delete_all_possible(self, words):
        words = words.strip()
        # if len(words.split(' ')) == 1:
        #     return words
        if 'to ' == words[:3]:
            words = words[3:]
        # words = words.replace('personx', '')
        # words = words.replace('persony', '')
        words = words.strip()
        tokens = self.nlp(words)
        res = []
        for token in tokens:
            # print('token', token, token.tag_)
            if 'V' in token.tag_:
                verb = self.verbs_all_tense(token.text)
                res.append(list(verb))
            # elif 'NN' in token.tag_ and 'char' not in token.text.lower():
            #     nouns = self.noun_all_tense(token.text)
            #     res.append(list(nouns))
            else:
                if 'PRP' in token.tag_: #or 'NNP' in token.tag_:
                    res.append([token.text, ''])
                else:
                    res.append([token.text])
            # print('res', res)
        all_words = []
        for tuple in [p for p in itertools.product(*res)]:
            words = ' '.join(tuple)
            if re.sub(r'[^\w]', ' ', words.strip()).strip() == 'none':
                pass
            else:
                all_words.append(re.sub(r'[^\w]', ' ', words.strip()).strip())
        return all_words

    def noun_adj_adv_find(self, words, prompt):
        """
        Try to find the noun, adj and adv for verbatlas outputs
        :param prompt: sentence
        :return:
        """
        tokens = self.nlp(prompt)
        res = []
        prev = ('', '')
        outputs = []
        for token in tokens:
            if 'NN' in token.tag_:
                if 'NN' not in prev[-1] and token.lemma_ in words:  # noun 前面不是noun 比如 beautiful girl
                    res.append(token.lemma_)
                elif 'NN' in prev[-1] and token.lemma_ in words:  # noun phrase
                    if res:
                        res[-1] += ' ' + token.lemma_
                    else:
                        res.append(token.lemma_)
                else:
                    pass
            if 'JJ' in prev[-1] or 'RB' in prev[-1]:
                if 'NN' in token.tag_:
                    outputs.append((token.lemma_, prev[0]))
                elif 'NN' not in token.tag_ and token.lemma_ in words:
                    res.append(token.lemma_)

            prev = (token.lemma_, token.tag_)

        return res, outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=str, default="apple", help="a obj")
    args = parser.parse_args()
    nltk_gen = nltk_gen()
    # prompt = args.example
    words = "saw"
    prompt = 'Susan is looking forward to it with Jim. She is very happy.'
    words = 'Susan is learning maths'
    # print(nltk_gen.verb_tense_gen('loves', "past", 3, "singular"))
    # print(nltk_gen.noun_adj_adv_find(words, prompt))
    # print(nltk_gen.comet_to_delete(words, prompt))
    # nltk_gen.read_tags(prompt)
    nltk_gen.verbs_all_tense('loves')
    nltk_gen.noun_all_tense('coffee')
    print(nltk_gen.comet_to_delete_all_possible(words='go to bed'))