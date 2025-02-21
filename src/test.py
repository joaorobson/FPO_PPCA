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

def preprocess_text(text):
    """Preprocesses Portuguese text by normalizing and removing stopwords."""

    text = re.sub(r'Art\. \d+(º?)(\.?)', '', text, flags=re.MULTILINE)
    text = re.sub(r'^TÍTULO M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})', '', text, flags=re.MULTILINE)
    text = re.sub(r'^CAPÍTULO M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Seção M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})', '', text, flags=re.MULTILINE)
    text = re.sub(r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}) – ', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Parágrafo único.', '', text, flags=re.MULTILINE)
    text = re.sub(r'^§ \d+(º?)', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\w\)', '', text, flags=re.MULTILINE)
    text = re.sub(r'\sM{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})(?=\s|$|\,|\.|\;)', '', text, flags=re.MULTILINE)

    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    text = re.sub(r'[^a-zA-Z\s\.\:\;]', '', text)

    words = re.split(r"[.;:]", text)

    stop_words = set(stopwords.words('portuguese'))
    stop_words.update(["lei", "leis", "previsto", "previstos", "inciso", "incisos", "paragrafo", "paragrafos",
                "art", "artigo", "caput", "termos", "salvo", "sobre", "desde"])

    print(stop_words)

    stop_words = [
        "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        for s in stop_words
    ]

    print(len(words))
    words = [word for word in words if word.strip() not in stop_words]

    return ' '.join(words)

f = open('cf/2024/titulo2.txt', 'r')

a = preprocess_text(f.read())

f.close()

print(a)
print(len(a.split()))
