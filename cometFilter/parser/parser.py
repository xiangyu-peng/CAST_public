from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPParser
import copy, pandas, os

# Neural Dependency Parser
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

# NER parser
ner_parser = CoreNLPParser(url='http://localhost:9000', tagtype='ner')

import spacy

nlp = spacy.load('en')

# Add neural coref to SpaCy's pipe
import neuralcoref

neuralcoref.add_to_pipe(nlp)

list_pronouns = ["he", "she", "it", "they", "him", "her", "them"]
import pickle

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

animate_list = pickle.load(open(os.path.join(__location__, 'animate.p'), 'rb'))


def is_animate(word):
    return word.lower() in animate_list


def parse_sentence(sent, verbose=False):
    """
    Parse a sentence and return the list of animate characters in that sentence, ideally.
    :param sent: The sentence to be parsed.
    :return: The list of characters.
    """
    # delete punctuation.
    sent = punc_remove(sent)

    # A list of characters
    set_char = set()

    # The dependency tree
    ner_list = ner_parser.tag(sent.split())
    if verbose:
        print("List of NER entities:", ner_list)
        print('-------------')

    sent_copy = []
    for word, tag in ner_list:
        if tag != 'PERSON' and tag != 'O' and tag != "TITLE":
            # sent_copy = sent_copy.replace(word, "[NER]")
            sent_copy.append("NER")
        elif tag == "TITLE" or tag == "PERSON":
            sent_copy.append(word)
            if word.lower() not in animate_list:
                animate_list.append(word.lower())
            set_char.add(word)
            print('add =>', word)
        elif tag == 'O':
            sent_copy.append(word)

    sent_copy = " ".join(sent_copy)
    print('sent_copy => ', sent_copy)
    print('animate_list =>', animate_list)
    dep_tree = dep_parser.raw_parse(sent_copy)
    dep_triples = list(list(dep_tree)[0].triples())
    dep_triples_copy = copy.copy(dep_triples)

    for governer, dep, dependent in dep_triples:
        if verbose:
            print(governer, dep, dependent)
        """
            Possible cases for being a character / subject.
            Since ROC stories is pretty simple in sentence structure, this should suffice.
        """
        if 'nsubj' in dep:
            if dependent[1] == 'PRP' or dependent[1] == 'PRP$' or 'NN' in dependent[1]:
                set_char.add(dependent[0])
        elif 'obj' in dep:
            if ('NN' == dependent[1] and is_animate(dependent[0])) or (dependent[1] == 'NNP'):
                set_char.add(dependent[0])
        elif dep == 'conj':
            if governer[0] in set_char and dependent[0] not in set_char:
                set_char.add(dependent[0])
            elif governer[0] not in set_char and dependent[0] in set_char:
                set_char.add(governer[0])
        elif dependent[1] == 'NNP':
            set_char.add(dependent[0])
        elif governer[1] == 'NNP':
            set_char.add(governer[0])

        if verbose:
            print('set_char => ', set_char)
    compounds = set()
    # Iterate through dependency again to check for nmod:poss
    # To make 'husband' --> 'my husband'
    for governer, dep, dependent in dep_triples_copy:
        if dep == 'nmod:poss' and governer[0] in set_char and governer[1] != 'NNP':
            set_char.remove(governer[0])
            if dependent[1] == 'PRP$':
                set_char.add(dependent[0] + " " + governer[0])
            else:
                set_char.add(dependent[0] + "\'s " + governer[0])
        elif dep == 'compound' and (governer[0] in set_char or dependent[0] in set_char):
            compounds.add(governer[0])
            compounds.add(dependent[0])
            set_char.add(governer[0])
            set_char.add(dependent[0])
    if verbose:
        print("Set of characters and Compounds ----")
        print(set_char)
        print(compounds)
        print("----------------")

    # Eliminate the compounds
    buffer = []
    for w in (sent + " .").split():
        if w in compounds:
            buffer.append(w)
        elif len(buffer):
            for b in buffer:
                if b in set_char:
                    set_char.remove(b)
            set_char.add(" ".join(buffer))
            buffer = []

    # Use coreference cluster
    doc = nlp(sent)
    print('doc', doc)
    """
        Sometimes both the main and the coref mentions are added to the list of characters.
        So let's do something about it.
    """
    for c in doc._.coref_clusters:
        if c.main.text in set_char:
            for m in c.mentions:
                if m.text in set_char and m.text != c.main.text:
                    set_char.remove(m.text)
    list_char = list(set_char)
    for c in set_char:
        if c.lower() == "ner" or c.lower() in list_pronouns:
            list_char.remove(c)
        elif c.lower() == "me" and "I" in set_char:
            list_char.remove("me")
        elif c.lower() == "myself" and "I" in set_char:
            list_char.remove("me")
    if verbose:
        print(list_char)

    #becky#
    # determine the rank of 2 characters.
    idx_1, idx_2 = 0, 0
    if len(list_char) == 2:
        print(list_char)
        if len(list_char[0].split()) == 1 and list_char[0] in sent.split():
            idx_1 = sent.split().index(list_char[0])
        else:
            for w in list_char[0].split():
                if w in sent.split():
                    idx_1 = max(idx_1, sent.split().index(w))

        if len(list_char[1].split()) == 1 and list_char[1] in sent.split():
            idx_2 = sent.split().index(list_char[1])
        else:
            for w in list_char[1].split():
                if w in sent.split():
                    idx_2 = max(idx_2, sent.split().index(w))


    if idx_1 > idx_2:
        list_char.reverse()

    return list_char


def parse_dataset(path):
    """
    Parse through a list of given sentences, generate two files.
    The first file is the parsing results of format (sentence, list of characters in the sentence).
    The second file contains sentences that have two or more characters.
    :param path: The path to the set of given sentences.
    """
    input_file = open(path, "r")
    input_lines = input_file.readlines()
    df_verify = pandas.DataFrame([], columns=["Sentence", "Parser Characters"])
    df_two = pandas.DataFrame([], columns=["Sentence", "Parser Characters"])
    for ind, l in enumerate(input_lines):
        characters = parse_sentence(l.strip()[:-1])
        df_new = pandas.DataFrame({"Sentence": [l.replace("\n", "")], "Parser Characters": ["; ".join(characters)]})
        df_verify = df_verify.append(df_new, ignore_index=True)
        if len(characters) >= 2:
            df_two = df_two.append(df_new, ignore_index=True)
    df_verify.to_csv("results.csv")
    df_two.to_csv("two_characters.csv")


import re
def punc_remove(text):
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@?[\\]^_`{|}~]+'
    return re.sub(remove_chars, '', text)

"""
    Usage example of parsing a specific sentence.
"""

import sys
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == "__main__":
    import csv
    with open('/home/becky/Documents/CommonsenseStoryGen/cometFilter/parser/two_characters.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            with HiddenPrints():
                r = parse_sentence(row[1], True)
            print(row[1])
            print(r)
"""
    Usage example of parsing a file with individual sentences.
"""
# parse_dataset("/media/COMET/ExamineNER/input_sentences.txt")