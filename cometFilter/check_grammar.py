from stanfordcorenlp import StanfordCoreNLP
from parser.parser import *

if __name__ == "__main__":
    sentence_1 = 'Anita kind returned their lunch whenever I.'
    ner_parser = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
    print(ner_parser.tag(sentence_1.split()))