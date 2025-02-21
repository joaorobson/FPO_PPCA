import os
import re
from collections import defaultdict
import math
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import unicodedata
from gensim.models import KeyedVectors

#model = KeyedVectors.load_word2vec_format('cbow_s100.txt')

def get_similarity(w1, w2):

    if w1 in model and w2 in model:
        return model.similarity(word1, word2)
    return 0


#https://ryanmcd.github.io/papers/globsumm.pdf
#https://aclanthology.org/C12-1056.pdf
#https://aclanthology.org/C10-2105.pdf

nltk.download('stopwords')
senadores = ['alan rick', 'alessandro vieira', 'ana paula lobato', 'angelo coronel', 'astronauta marcos pontes', 'augusta brito', 'beto faro', 'carlos portinho', 'carlos viana', 'chico rodrigues', 'cid gomes', 'ciro nogueira', 'cleitinho', 'confucio moura', 'damares alves', 'daniella ribeiro', 'davi alcolumbre', 'dr. hiran', 'dra. eudocia', 'eduardo braga', 'eduardo girao', 'eduardo gomes', 'efraim filho', 'eliziane gama', 'esperidiao amin', 'fabiano contarato', 'fernando dueire', 'fernando farias', 'flavio arns', 'flavio bolsonaro', 'giordano', 'hamilton mourao', 'humberto costa', 'iraja', 'ivete da silveira', 'izalci lucas', 'jader barbalho', 'jaime bagattoli', 'jaques wagner', 'jayme campos', 'jorge kajuru', 'jorge seif', 'jussara lima', 'laercio oliveira', 'leila barros', 'lucas barreto', 'luis carlos heinze', 'magno malta', 'mara gabrilli', 'marcelo castro', 'marcio bittar', 'marcos rogerio', 'marcos do val', 'margareth buzetti', 'mecias de jesus', 'nelsinho trad', 'omar aziz', 'oriovisto guimaraes', 'otto alencar', 'paulo paim', 'plinio valerio', 'professora dorinha seabra', 'randolfe rodrigues', 'renan calheiros', 'rodrigo pacheco', 'rogerio marinho', 'rogerio carvalho', 'romario', 'sergio moro', 'soraya thronicke', 'styvenson valentim', 'sergio petecao', 'teresa leitao', 'tereza cristina', 'vanderlan cardoso', 'veneziano vital do rego', 'wellington fagundes', 'weverton', 'wilder morais', 'zenaide maia', 'zequinha marinho']


def preprocess_text(text):
    """Preprocesses Portuguese text by normalizing and removing stopwords."""

    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    text = re.sub(r'[^a-zA-Z\s]', '', text)
    for senador in senadores:
        text = re.sub(r'\b' + re.escape(senador) + r'\b', '', text)

    words = text.split()

    stop_words = set(stopwords.words('portuguese'))
    stop_words.update(["senador", "senadora", "gabinete", "constituicao", "federal", "art", 
    "constitucional", "emenda", "senado", "nao" ])
    words = [word for word in words if word not in stop_words and len(word) > 2]

    return ' '.join(words)

def tokenize(text):
    """Tokenizes the input text into words."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def calculate_tf(tokens):
    """Calculates term frequency for each bigram."""
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    return tf

def calculate_idf(documents_count, bag_of_words):
    """Calculates the inverse document frequency for each bigram."""
    idf = {}
    for token, count in bag_of_words.items():
        idf[token] = math.log(documents_count / count)
    return idf

def main(base_folder):
    year_tokens = defaultdict(list)
    bag_of_words = defaultdict(int)
    documents_count = 0 


    # Read documents and generate bow
    for year in os.listdir(base_folder):
        documents_count += 1
        year_path = os.path.join(base_folder, year)
        if os.path.isdir(year_path):
            for filename in tqdm(os.listdir(year_path)):
                if filename.endswith('.txt'):
                    with open(os.path.join(year_path, filename), 'r', encoding='utf-8') as file:
                        text = preprocess_text(file.read())
                        tokens = tokenize(text)
                        year_tokens[year].extend(tokens)

            # Create bag-of-words 
            unique_tokens = set(year_tokens[year])
            for token in unique_tokens:
                bag_of_words[token] += 1

    # Calculate TF for each year
    year_tf = defaultdict(int) 
    for year, tokens in year_tokens.items():
        year_tf[year] = calculate_tf(tokens)

    idf = calculate_idf(documents_count, bag_of_words)
        
    # Calculate TF-IDF for each bigram in each year
    year_tfidf = defaultdict(dict)
    for year, freqs in year_tf.items():
        for token, freq in freqs.items():
            year_tfidf[year][token] = freq * (idf[token] + 1)

    return year_tfidf


if __name__ == "__main__":
    base_folder = 'txts'  # Replace with your path
    tfidf_results = main(base_folder)
    print(tfidf_results)
    for year, tfidf in tfidf_results.items():
        print(year, sorted(tfidf.items(), key=lambda x:x[1], reverse=1)[:5])
