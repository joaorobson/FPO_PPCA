from tfidf import TFIDF
from ilp import SummaryGenerator
import itertools
from gensim.models import KeyedVectors
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Fonte: http://nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc
model = KeyedVectors.load_word2vec_format('cbow_s100.txt')

def get_sentence_vector(sentence):
    words = sentence.split()
    word_vectors = [model[word] for word in words if word in model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

def get_similarity(s1, s2):
    w1v = get_sentence_vector(s1)
    w2v = get_sentence_vector(s2) 

    return cosine_similarity([w1v], [w2v])


def get_similarities(words):
    s = {}
    for i, j in itertools.combinations(words, 2):
        s[(i, j)] = get_similarity(i, j)

    return s

if __name__ == "__main__":
    import time

    tfidf = TFIDF()
    summary_generator = SummaryGenerator()
    tfidf_results = tfidf.run("cf")
    sent_rev = tfidf.get_sent_relevance("cf", tfidf_results)

    sims = {}

    summaries = {}
    for year, docs in sent_rev.items():
        summaries[year] = {}
        for doc, tfidf in tqdm(docs.items()):
            print(year, doc)
            com = time.time()
            lengths = {i: len(i) for i in tfidf.keys()}
            s = get_similarities(list(tfidf.keys()))
            summ = summary_generator.get_summary(tfidf, s, lengths, 100)
            summaries[year][doc] = summ

    with open('summaries.json', 'w') as f:
        json.dump(summaries, f)
