from sentence_transformers import SentenceTransformer, util
import numpy as np

embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def similarity_detect(corpus, queries, threshold=0.8):
    if 'none' in corpus:
        corpus.pop(corpus.index('none'))
    if 'none' in queries:
        queries.pop(queries.index('none'))
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()
        for idx, score in enumerate(cos_scores):
            if score >= threshold:
                # print('Match!!!!')
                # print(corpus[idx])
                # print(query)
                # print(score)
                return True
    return False


    #We use np.argpartition, to only partially sort the top_k results
    # top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    # print("\n\n======================\n\n")
    # print("Query:", query)
    # print("\nTop 5 most similar sentences in corpus:")

    # for idx in top_results[0:top_k]:
    # for idx in range(5):
    #     print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))

def similarity_score_max(corpus, queries):
    if 'none' in corpus:
        corpus.pop(corpus.index('none'))
    if 'none' in queries:
        queries.pop(queries.index('none'))
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    scores_max = float('-inf')
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()
        for idx, score in enumerate(cos_scores):
            scores_max = max(float(scores_max), float(score))
    return scores_max

if __name__ == "__main__":
    corpus = ['to cry', 'sad', 'to be happy', 'love', 'to dream']
    queries = ['cry', 'to get way']
    # corpus = ['[Char_1] was a cop with a police station.']
    # queries = ['[Char_1] was a cop in a police station']
    # print(similarity_detect(corpus, queries))
    print(similarity_score_max(corpus, queries))
