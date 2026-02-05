import ssl
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy3
from nltk.corpus import stopwords
import nltk
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import numpy as np



import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('stopwords')

CSV_FILE = "songs.csv"

df = pd.read_csv(CSV_FILE)

morph = pymorphy3.MorphAnalyzer()
russian_stopwords = set(stopwords.words("russian"))
english_stopwords = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^а-яё\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in russian_stopwords and w not in english_stopwords]
    words = [morph.parse(w)[0].normal_form for w in words]
    return " ".join(words)

df["processed_text"] = df["text"].apply(preprocess_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["processed_text"])
tfidf_feature_names = vectorizer.get_feature_names_out()

tfidf_sum = tfidf_matrix.sum(axis=0)
tfidf_scores = [(word, tfidf_sum[0, idx]) for idx, word in enumerate(tfidf_feature_names)]
tfidf_scores.sort(key=lambda x: x[1], reverse=True)

print("Топ-20 слов по TF-IDF:")
for word, score in tfidf_scores[:20]:
    print(f"{word}: {score:.4f}")

all_text = " ".join(df["processed_text"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


sentences = [text.split() for text in df["processed_text"]]

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1,
    epochs=50
)

test_word = "грех"  
if test_word in w2v_model.wv:
    print(f"Слова близкие к слову '{test_word}':")
    for word, score in w2v_model.wv.most_similar(test_word, topn=5):
        print(f"{word}: {score:.3f}")
else:
    print(f"Слово '{test_word}' отсутствует  в словаре модели")

word_counts = Counter(all_text.split())
top_words = [word for word, _ in word_counts.most_common(15)
             if word in w2v_model.wv]

vectors = np.array([w2v_model.wv[word] for word in top_words])

tsne = TSNE(
    n_components=2,
    perplexity=5,
    init="random"
)

coords = tsne.fit_transform(vectors)

plt.figure(figsize=(8, 8))
for i, word in enumerate(top_words):
    plt.scatter(coords[i, 0], coords[i, 1])
    plt.text(coords[i, 0] + 0.3, coords[i, 1] + 0.3, word)

plt.title("t-SNE Word2Vec (по 15 частых словам)")
plt.show()
