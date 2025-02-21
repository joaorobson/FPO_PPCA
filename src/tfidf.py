import os
import sys
import re
from collections import defaultdict
import math
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import unicodedata
from gensim.models import KeyedVectors

#model = KeyedVectors.load_word2vec_format('cbow_s100.txt')



#https://ryanmcd.github.io/papers/globsumm.pdf
#https://aclanthology.org/C12-1056.pdf
#https://aclanthology.org/C10-2105.pdf

nltk.download('stopwords')

class TFIDF:
    def preprocess_text(self, text):

        text = re.sub(r'Art\. \d+(º?)(\.?)', '', text, flags=re.MULTILINE)
        text = re.sub(r'^TÍTULO M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})', '', text, flags=re.MULTILINE)
        text = re.sub(r'^CAPÍTULO M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Seção M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})', '', text, flags=re.MULTILINE)
        text = re.sub(r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}) – ', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Parágrafo único.', '', text, flags=re.MULTILINE)
        text = re.sub(r'^§ \d+(º?)', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\w\)', '', text, flags=re.MULTILINE)
        text = re.sub(r'&quot;', '', text, flags=re.MULTILINE)
        #text = re.sub(r'\sM{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})(?=\s|$|\,|\.|\;)', '', text, flags=re.MULTILINE)


        text = text.lower()
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        
        text = re.sub(r'[.;:]$', ' &&&& ', text,  flags=re.MULTILINE)
        text = re.sub(r'\n', ' &&&& ', text,  flags=re.MULTILINE)
        #print('---------------', text)
        text = re.sub(r'[^a-zA-Z\s\&]', '', text)

        words = text.split()


        stop_words = set(stopwords.words('portuguese'))
        stop_words.update(["lei", "leis", "previsto", "previstos", "inciso", "incisos", "paragrafo", "paragrafos",
                        "art", "artigo", "caput", "termos", "salvo", "sobre", "desde", "alinea"])

        stop_words = [
            "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
            for s in stop_words
        ]

        words = [word for word in words if word not in stop_words]

        return ' '.join(words)

    def sent_tokenize(self, text):
        text = text.lower()
        tokens = re.split("&&&&", text)
        return [t.strip() for t in tokens if t.strip()]

    def tokenize(self, text):
        """Tokenizes the input text into words."""
        text = text.lower()
        text = re.sub(r'[\.\;\:\&]', '', text)
        tokens = re.findall(r'\b\w+\b', text)
        #tokens = re.split(r"[.;:]", text)
        #tokens = [t.strip() for t in tokens]
        
        return tokens

    def calculate_tf(self, tokens):
        """Calculates term frequency for each bigram."""
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        return tf

    def calculate_idf(self, documents_count, bag_of_words):
        """Calculates the inverse document frequency for each bigram."""
        idf = {}
        for token, count in bag_of_words.items():
            idf[token] = math.log(documents_count / count)
        return idf

    def get_sent_relevance(self, base_folder, tfidf):
        sent_rev = defaultdict(dict)
        for year in os.listdir(base_folder):
            year_path = os.path.join(base_folder, year)
            if os.path.isdir(year_path):
                for filename in tqdm(os.listdir(year_path)):
                    tit = re.findall('(.*)\..*', filename)[0]
                    sent_rev[year][tit] = {}
                    if filename.endswith('.txt'):
                        with open(os.path.join(year_path, filename), 'r', encoding='utf-8') as f:
                            #print(year, filename)
                            text = self.preprocess_text(f.read())
                            sents = self.sent_tokenize(text)
                            #print(len(sents))
                            for sent in sents:
                                sent_rev[year][tit][sent] = sent_rev[year][tit].get(sent, 0)
                                for token in sent.split():
                                    sent_rev[year][tit][sent] += tfidf[year][tit][token]
        return sent_rev
        
    def run(self, base_folder):
        year_tokens = defaultdict(dict)
        bag_of_words = defaultdict(dict)
        documents_count = defaultdict(int) 


        # Read documents and generate bow
        for year in os.listdir(base_folder):
            year_path = os.path.join(base_folder, year)
            if os.path.isdir(year_path):
                for filename in tqdm(os.listdir(year_path)):
                    tit = re.findall('(.*)\..*', filename)[0]
                    year_tokens[year][tit] = []
                    if filename.endswith('.txt'):
                        documents_count[year] += 1
                        with open(os.path.join(year_path, filename), 'r', encoding='utf-8') as file:
                            text = self.preprocess_text(file.read())
                            tokens = self.tokenize(text)
                            year_tokens[year][tit].extend(tokens)

                            # Create bag-of-words 
                            unique_tokens = set(tokens)
                            for token in unique_tokens:
                                bag_of_words[year][token] = bag_of_words[year].get(token, 0) + 1

        # Calculate TF for each year
        year_tf = defaultdict(dict) 
        for year, docs in year_tokens.items():
            for doc, tokens in docs.items():
                year_tf[year][doc] = self.calculate_tf(tokens)

        idf = defaultdict(dict) 
        for year, docs in year_tokens.items():
            idf[year] = self.calculate_idf(documents_count[year], bag_of_words[year])
            
        # Calculate TF-IDF for each bigram in each year
        year_tfidf = defaultdict(dict)
        for year, docs in year_tf.items():
            for doc, freqs in docs.items():
                year_tfidf[year][doc] = {}
                for token, freq in freqs.items():
                    year_tfidf[year][doc][token] = freq * (idf[year][token] + 1)

        return year_tfidf


if __name__ == "__main__":
    base_folder = 'cf'  # Replace with your path
    tfidf = TFIDF()
    tfidf_results = tfidf.run(base_folder)
    sent_rev = tfidf.get_sent_relevance(base_folder, tfidf_results) 


    
    #for year, docs in tfidf_results.items():
    #    for doc, tfidf in docs.items():
    #        print(year, doc, sorted(tfidf.items(), key=lambda x:x[1], reverse=1)[:10])
