import pandas
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords

stopwords = stopwords.words('english')
pickle.dump(stopwords, open("/home/becky/Documents/CommonsenseStoryGen/cometFilter/beam_search/stopwords.p", "wb+"))
print("DONE")
input()

roc_df = pandas.read_csv("/home/becky/Documents/CommonsenseStoryGen/cometFilter/beam_search/ROCStories_winter2017 - ROCStories_winter2017.csv")
list_sent = []
from sklearn.feature_extraction.text import TfidfVectorizer

for col in range(1, 6):
    list_sent.extend(list(roc_df['sentence' + str(col)]))

vectorizer = TfidfVectorizer()
vectorizer.fit(list_sent)
# print(vectorizer.vocabulary_)
print(len(vectorizer.vocabulary_))
vocab = vectorizer.vocabulary_
idf = vectorizer.idf_

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(list_sent)
vocab_tf = vectorizer.get_feature_names()
print(vectorizer.get_stop_words())
input()
voc_tf = dict()
for i, word in enumerate(vocab_tf):
    voc_tf.update({word: np.sum(X[i])})

voc_idf = dict()
for w in vocab:
    voc_idf.update({w: idf[vocab[w]]})

tf_idf = dict()
for word in vocab:
    tf_idf.update({word: voc_tf[word] * voc_idf[word]})

pickle.dump(tf_idf, open("voc_idf.p", "wb+"))