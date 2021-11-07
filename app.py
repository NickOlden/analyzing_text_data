import nltk
import string
import gensim
from texts.text1 import txt
from texts.text2 import corpus
from nltk.text import TextCollection
from sklearn.preprocessing import Binarizer
from collections import Counter, defaultdict
from constants.gender import MALE_WORDS, FEMALE_WORDS
from constants.gender import MALE, FEMALE, UNKNOWN, BOTH
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def parse_gender(text):
    """
    @text
    """
    sentences = [
        [word.lower() for word in nltk.word_tokenize(sentence)]
        for sentence in nltk.sent_tokenize(text)
    ]
    sents, words = count_gender(sentences)
    total = sum(words.values())
    for gender, count in words.items():
        pcent = (count / total) * 100
        nsents = sents[gender]
        yield f"{round(pcent, 2)}% {gender} ({nsents} sentences)"


def count_gender(sentences):
    """
    @sentences
    """
    sents = Counter()
    words = Counter()
    for sentence in sentences:
        gender = genderize(sentence)
        sents[gender] += 1
        words[gender] += len(sentence)
    return sents, words


def genderize(words):
    """
    @words
    """
    mwlen = len(MALE_WORDS.intersection(words))
    fwlen = len(FEMALE_WORDS.intersection(words))
    if mwlen > 0 and fwlen == 0:
        return MALE
    elif mwlen == 0 and fwlen > 0:
        return FEMALE
    elif mwlen > 0 and fwlen > 0:
        return BOTH
    else:
        return UNKNOWN


def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()
    for token in nltk.word_tokenize(text):
        if token in string.punctuation:
            continue
        yield stem.stem(token)


def vectorize(doc):
    features = defaultdict(int)
    for token in tokenize(doc):
        features[token] += 1
    return features


def vectorize_corpus(corpus):
    corpus = [tokenize(doc) for doc in corpus]
    texts = TextCollection(corpus)
    for doc in corpus:
        yield {
            term: texts.tf_idf(term, doc)
            for term in doc
        }


if __name__ == "__main__":
    delimetr = "+++++++++"

    for x in parse_gender(txt):
        print(x)
    print(delimetr)

    vectors = map(vectorize, corpus)
    for x in vectors:
        print(x)
    print(delimetr)

    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    for x in vectors:
        print(x)
    print(delimetr)

    c = [tokenize(doc) for doc in corpus]
    id2word = gensim.corpora.Dictionary(c)
    vectors = [
        id2word.doc2bow(doc) for doc in c
    ]
    for x in vectors:
        print(x)
    print(delimetr)

    freq = CountVectorizer()
    c = freq.fit_transform(corpus)
    onehot = Binarizer()
    c = onehot.fit_transform(c.toarray())
    for x in c:
        print(x)
    print(delimetr)

    c = [tokenize(doc) for doc in corpus]
    id2word = gensim.corpora.Dictionary(c)
    vectors = [
        [(token[0], 1) for token in id2word.doc2bow(doc)]
        for doc in c
    ]
    for x in vectors:
        print(x)
    print(delimetr)

    tfidf = TfidfVectorizer()
    c = tfidf.fit_transform(corpus)
    for x in c:
        print(x)
    print(delimetr)

    c = [tokenize(doc) for doc in corpus]
    lexicon = gensim.corpora.Dictionary(c)
    tfidf = gensim.models.TfidfModel(dictionary=lexicon, normalize=True)
    vectors = [tfidf[lexicon.doc2bow(doc)] for doc in c]
    for x in vectors:
        print(x)
    lexicon.save_as_text('temp/lexicon.txt', sort_by_word=True)  # save
    tfidf.save('temp/tfidf.pkl')  # save
    lexicon = gensim.corpora.Dictionary.load_from_text('lexicon.txt')  # load
    tfidf = gensim.models.TfidfModel.load('tfidf.pkl')  # load

    print(delimetr)
