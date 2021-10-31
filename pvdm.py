import re
from gensim.models import word2vec
import nltk
from nltk.corpus import stopwords
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

paragraph = """
        As in all machine-learning research, we assume we have at least two and
        preferably three sets of problem examples. The first is the training set. It
        is used to adjust the parameters of the model. The second is called the
        development set and is used to test the model as we try to improve it. (It
        is also referred to as the held-out set or the validation set.) The third is
        the test set. Once the model is fixed and (if we are lucky) producing good
        results, we then evaluate on the test-set examples. This prevents us from
        accidentally developing a program that works on the development set but
        not on yet unseen problems. These sets are sometimes called corpora, as
        in “test corpus.” The Mnist data we use is available on the web. The
        training data consists of 60,000 images and their correct labels, and the
        development/test set has 10,000 images and labels.
        The great property of the perceptron algorithm is that, if there is a set
        of parameter values that enables the perceptron to classify all the training
        set correctly, the algorithm is guaranteed to find it. Unfortunately, for most
        real-world examples there is no such set. On the other hand, even then
        perceptrons often work remarkably well in the sense that there are parameter
        settings that label a very high percentage of the examples correctly."""

# Preprocessing the data
text = re.sub(r'\[[0-9]*\]', ' ', paragraph)
text = re.sub(r'\s+', ' ', text)
text = text.lower()
text = re.sub(r'\d', ' ', text)
text = re.sub(r'\s+', ' ', text)

# preprocessing of the dataset
sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

# model = Doc2Vec(documents=sentences, corpus_file=None, vector_size=100, dm_mean=None, dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1, dv=None, dv_mapfile=None, comment=None, trim_rule=None, callbacks=(), window=5, epochs=10)
# Training the Word2vec model
model = Word2Vec(sentences, min_count=1)

words = model.wv.vocab

# Finding the word vector
vector= model.wv['research']
print(vector)

# Most similar words
similar = model.wv.most_similar('research')
print('Similar words', similar)
