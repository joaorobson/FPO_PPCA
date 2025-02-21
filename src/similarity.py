from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('cbow_s100.txt')

word1 = "rei"
word2 = "rainhakk"

if word1 in model and word2 in model:
    similarity = model.similarity(word1, word2)
    print(f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}")
else:
    print("One or both words are not in the vocabulary.")

