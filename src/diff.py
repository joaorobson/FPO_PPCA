from difflib import SequenceMatcher

def similarity_ratio(doc1, doc2):
    return SequenceMatcher(None, doc1, doc2).ratio()

def text_similarity(doc1, doc2):
    words1 = set(doc1.split())
    words2 = set(doc2.split())

    intersection = words1 & words2  # Words in both
    added = words2 - words1         # New words in doc2
    removed = words1 - words2       # Words missing from doc1 to doc2

    return {
        "common_words": intersection,
        "added_words": added,
        "removed_words": removed,
        "change_ratio": len(intersection) / max(len(words1 | words2), 1)  # Jaccard similarity
    }

# Example Usage:

for i in range(1, 9):

    f = open(f'cf/1988/titulo{i}.txt','r')
    text1 = f.read()
    f.close()

    f = open(f'cf/2024/titulo{i}.txt','r')
    text2 = f.read()
    f.close()



    #similarity = similarity_ratio(text1, text2)
    #print(f"Text similarity: {similarity:.2f}")

    similarity = text_similarity(text1, text2)
    #print("Words in common:", similarity["common_words"])
    #print("Words added:", similarity["added_words"])
    #print("Words removed:", similarity["removed_words"])
    print(f"{i} - Similarity Ratio: {similarity['change_ratio']:.2f}")
