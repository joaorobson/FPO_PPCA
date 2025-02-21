import os
import re
import math
import unicodedata
from collections import defaultdict
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.util import bigrams

nltk.download('stopwords')
senadores = ['alan rick', 'alessandro vieira', 'ana paula lobato', 'angelo coronel', 'astronauta marcos pontes', 'augusta brito', 'beto faro', 'carlos portinho', 'carlos viana', 'chico rodrigues', 'cid gomes', 'ciro nogueira', 'cleitinho', 'confucio moura', 'damares alves', 'daniella ribeiro', 'davi alcolumbre', 'dr. hiran', 'dra. eudocia', 'eduardo braga', 'eduardo girao', 'eduardo gomes', 'efraim filho', 'eliziane gama', 'esperidiao amin', 'fabiano contarato', 'fernando dueire', 'fernando farias', 'flavio arns', 'flavio bolsonaro', 'giordano', 'hamilton mourao', 'humberto costa', 'iraja', 'ivete da silveira', 'izalci lucas', 'jader barbalho', 'jaime bagattoli', 'jaques wagner', 'jayme campos', 'jorge kajuru', 'jorge seif', 'jussara lima', 'laercio oliveira', 'leila barros', 'lucas barreto', 'luis carlos heinze', 'magno malta', 'mara gabrilli', 'marcelo castro', 'marcio bittar', 'marcos rogerio', 'marcos do val', 'margareth buzetti', 'mecias de jesus', 'nelsinho trad', 'omar aziz', 'oriovisto guimaraes', 'otto alencar', 'paulo paim', 'plinio valerio', 'professora dorinha seabra', 'randolfe rodrigues', 'renan calheiros', 'rodrigo pacheco', 'rogerio marinho', 'rogerio carvalho', 'romario', 'sergio moro', 'soraya thronicke', 'styvenson valentim', 'sergio petecao', 'teresa leitao', 'tereza cristina', 'vanderlan cardoso', 'veneziano vital do rego', 'wellington fagundes', 'weverton', 'wilder morais', 'zenaide maia', 'zequinha marinho']


def preprocess_text(text):
    """Preprocesses Portuguese text by normalizing, removing stopwords, and keeping only letters."""
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    for senador in senadores:
        text = re.sub(r'\b' + re.escape(senador) + r'\b', '', text)


    words = text.split()
    stop_words = set(stopwords.words('portuguese'))
    stop_words.update(["senador", "senadora", "gabinete"])

    words = [word for word in words if word not in stop_words and len(word) > 1]

    return words  # Returns list instead of string, useful for bigram creation

def tokenize(text):
    """Tokenizes text into bigrams."""
    words = preprocess_text(text)
    return list(bigrams(words))  # Generates bigrams

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
        idf[token] = math.log(documents_count / (count + 1))  # Avoid division by zero
    return idf

def main(base_folder):
    year_tokens = defaultdict(dict)
    bag_of_words = defaultdict(dict)
    documents_count = defaultdict(int) 

    # Read documents and generate bag-of-bigrams
    for year in os.listdir(base_folder):
        year_path = os.path.join(base_folder, year)
        if os.path.isdir(year_path):
            for filename in tqdm(os.listdir(year_path)):
                pec = re.findall(r'(.*)-.*', filename)[0]
                year_tokens[year][pec] = []
                if filename.endswith('.txt'):
                    documents_count[year] += 1
                    with open(os.path.join(year_path, filename), 'r', encoding='utf-8') as file:
                        text = file.read()
                        tokens = tokenize(text)
                        year_tokens[year][pec].extend(tokens)

                        # Create bag-of-bigrams 
                        unique_tokens = set(tokens)
                        for token in unique_tokens:
                            bag_of_words[year][token] = bag_of_words[year].get(token, 0) + 1

    # Calculate TF for each year
    year_tf = defaultdict(dict) 
    for year, docs in year_tokens.items():
        for doc, tokens in docs.items():
            year_tf[year][doc] = calculate_tf(tokens)

    idf = defaultdict(dict) 
    for year, docs in year_tokens.items():
        idf[year] = calculate_idf(documents_count[year], bag_of_words[year])
        
    # Calculate TF-IDF for each bigram in each year
    year_tfidf = defaultdict(dict)
    for year, docs in year_tf.items():
        for doc, freqs in docs.items():
            year_tfidf[year][doc] = {}
            for token, freq in freqs.items():
                year_tfidf[year][doc][token] = freq * (idf[year][token] + 1)

    return year_tfidf

if __name__ == "__main__":
    base_folder = 'txts'  # Replace with your path
    tfidf_results = main(base_folder)
    for year, docs in tfidf_results.items():
        for doc, tfidf in docs.items():
            print(year, doc, sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:5])

