# from stanfordcorenlp import StanfordCoreNLP
# from parser.parser import *
import spacy
import neuralcoref
from collections import defaultdict
nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

# First solution, filter out all the sentences with third char
def filter_third_char(text, chars):
    doc = nlp(text)
    if doc._.has_coref:
        doc_sol = nlp(doc._.coref_resolved.split('.')[1])  # Unicode representation of the doc where each corefering mention is replaced by the main mention in the associated cluster.
        for i, token in enumerate(doc_sol):
            if token.pos_ == 'PROPN' or token.pos_ == 'PRON':
                if str(token) not in chars:
                    return False
                    # print(i, str(token), token.pos_)
    return True  # it is ok to use this sentence

def coref_read(text):
    """
    Find name and then use CR to find all the index
    :param text:
    :return:
    """
    doc = nlp(text)
    names_idx = defaultdict(set)

    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

    for i, token in enumerate(doc):
        if token.text in persons:
            names_idx[token.text[0].upper() + token.text[1:].lower()].add(i)

    if doc._.has_coref:
        # print('>>>', doc._.coref_clusters[0].mentions[0].start, doc._.coref_clusters[0].mentions[1].start, doc._.coref_clusters[0].mentions[2].start)
        for idx in range(len(doc._.coref_clusters)):
            name = str(doc._.coref_clusters[idx].main)
            # print(name, names_idx, name in names_idx)
            if name in names_idx:
                for j in range(len(doc._.coref_clusters[idx].mentions)):
                    # print('j', j, doc._.coref_clusters[idx].mentions[j].start)
                    names_idx[name].add(doc._.coref_clusters[idx].mentions[j].start)

    # print('names_idx', names_idx)
    return names_idx


# Take into a dict() and return those sentence as a dict() which pass the filter
def coref_filter(gen_dict, chars, prev_text, search_time):
    output_dict = dict()
    id_new = 0
    for id in range(1, len(gen_dict)+1):
        text = gen_dict[id]
        if filter_third_char(prev_text + ' ' + text, chars):
            print('prev_text + text', prev_text + ' ' + text)
            id_new += 1
            output_dict[id_new] = text
            if id_new >= search_time:
                return output_dict
    return output_dict

if __name__ == "__main__":
    # nlp = StanfordCoreNLP(r'../../stanford-corenlp-4.1.0')
    # sentence = "Olivia went out with Harry on a date. Harry was expecting Maggie to come over to help him get home."
    # sentence_1 = "Johnny invited me back as well."
    # sentence_2 = "Harry was expecting Maggie to come over to help him get home."
    # print('Tokenize:', nlp.coref(sentence))
    # print(parse_sentence(sentence_1))
    # ner_parser = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
    # print(ner_parser.tag(sentence_1.split()))
    # print(parse_sentence(sentence_2))
    # print(nlp.ner(sentence_2))
    # nlp.close()

    # print(replace_third_char(text='Barry really likes his neighbor Sara. Sara knows him and loves to play with them.',
    #                    chars=['Barry', 'Sara']))
    gen_dict = dict()
    gen_dict[0] = 'Sara knows him and loves to play with them.'
    # gen_dict[1] = 'Sara knows him and loves to play with him.'
    # prev_text = 'Barry really likes his neighbor Sara.'
    # chars = ['Barry', 'Sara']
    # print(coref_filter(gen_dict=gen_dict, chars=chars, prev_text=prev_text, search_time=1))
    coref_read('Susan is looking forward to spending time with Jim. She is very happy to be with him. She loves it.')




