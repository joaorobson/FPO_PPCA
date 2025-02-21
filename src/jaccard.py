from tfidf import TFIDF
import json


t = TFIDF()
s = json.load(open('summaries.json','r'))

diffs = {}

for i in range(1, 9):
    j1988 = s['1988'][f'titulo{i}']
    j2024 = s['2024'][f'titulo{i}']

    f1988 = open(f'cf/1988/titulo{i}.txt')
    f2024 = open(f'cf/2024/titulo{i}.txt')
    text1988 = t.preprocess_text(f1988.read())
    text2024 = t.preprocess_text(f2024.read())
    sents1988 = set(t.sent_tokenize(text1988))
    sents2024 = set(t.sent_tokenize(text2024))
    intersection = sents1988 & sents2024  # Words in both
    added = sents2024 - sents1988 # New words in doc2
    removed = sents1988 - sents2024       # Words missing from doc1 to doc2
    print(len(added), len(sents2024), len(added)/len(sents2024), len(j2024), len(added & set(j2024))/len(j2024))
    print('--------------', added & set(j2024))
    diffs[i] = {
        "common_words": intersection,
        "added_words": added,
        "removed_words": removed,
        "change_ratio": len(intersection) / max(len(sents1988 | sents2024), 1)  # Jaccard similarity
    }
for i in diffs:
    print(i, f"{diffs[i]['change_ratio']:.2f}")

