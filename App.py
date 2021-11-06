import nltk
from collections import Counter
from texts.text1 import txt
from constants.gender import MALE, FEMALE, UNKNOWN, BOTH, MALE_WORDS, FEMALE_WORDS


def parse_gender(text):
    sentences = [
        [word.lower() for word in nltk.word_tokenize(sentence)]
        for sentence in nltk.sent_tokenize(text)
    ]
    sents, words = count_gender(sentences)
    total = sum(words.values())
    for gender, count in words.items():
        pcent = round((count / total) * 100)
        nsents = sents[gender]
        print(
            f"{pcent}% {gender} ({nsents} sentences)"
        )


def count_gender(sentences):
    sents = Counter()
    words = Counter()
    for sentence in sentences:
        gender = genderize(sentence)
        sents[gender] += 1
        words[gender] += len(sentence)
    return sents, words


def genderize(words):
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


def gen(i):
    for x in range(i):
        a = x * 2
        print(f'gen: {a}')
        yield a


if __name__ == "__main__":
    # parse_gender(txt)
    for x in gen(3):
        print(f'main: {x + 1}')
