import numpy as np, nltk
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
import string

wnl = nltk.WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def words_tokens(text, ignore_punct=False):
    if(ignore_punct):
        text = text.translate(None, string.punctuation)

    return [t for t in nltk.word_tokenize(text)]

def words_lems(text, lower=False, ignore_punct=False):
    text_pos = nltk.pos_tag(words_tokens(text, ignore_punct=ignore_punct))
    text_lems = [wnl.lemmatize(t,pos=get_wordnet_pos(p)) for t,p in text_pos]

    if lower:
        return [lem.lower() for lem in text_lems]
    else:
        return text_lems

def stem_word(word, stemmer, lower=False):
    """
    Tries to stem a word using the provided stemmer. If an error occurs (happens sometimes with the arabic stemmer), return the word as is
    """
    try:
        if lower:
            return stemmer.stem(word).lower()
        else:
            return stemmer.stem(word)
    except Exception:
        if lower:
            return word.lower()
        else:
            return word

def words_stems(text, lang="english", lower=False, ignore_stopwords=False, ignore_punct=False):
    tokens = words_tokens(text, ignore_punct=ignore_punct)
    stemmer = SnowballStemmer(lang, ignore_stopwords=ignore_stopwords)

    return [stem_word(t, stemmer, lower) for t in tokens]

def words_to_int(words, first_index=0, ignore_punct=False):
    if(ignore_punct):
        words_set = set([w for w in words if w not in string.punctuation])
    else:
        words_set = set(words)

    return {w:i for w,i in zip(words_set, range(first_index, first_index + len(words_set)))}

stopwords = nltk.corpus.stopwords.words('english')

def build_link(adj, weight, words_map, words, from_index, to_index, max_dist, links_to_stopwords=True):
    words_len = len(words)
    while to_index < words_len and (words[to_index] in string.punctuation or (not links_to_stopwords and words[to_index] in stopwords)):
        to_index += 1
        weight /= 2

    if (to_index - from_index) <= max_dist and to_index < len(words):
        adj[words_map[words[from_index]], words_map[words[to_index]]] = adj[words_map[words[from_index]], words_map[words[to_index]]] + weight
    weight /= 2
    return weight, to_index + 1

def build_graph(lemmas, lemmas_map, max_dist=4, max_weight=16, links_from_stopwords=True, links_to_stopwords=True):
    len_dist_lemmas = len(lemmas_map)
    len_lemmas = len(lemmas)
    adj = np.zeros((len_dist_lemmas, len_dist_lemmas))
    for index, lemma in enumerate(lemmas):
        # TODO Take into account stop words
        if lemma in string.punctuation or (not links_from_stopwords and lemma in stopwords):
            continue
        weight = max_weight
        next_index = index + 1

        for i in range(0, max_dist):
            weight, next_index = build_link(adj, weight, lemmas_map, lemmas, index, next_index, max_dist, links_to_stopwords)

    return adj

def text_to_graph(text, normalization="lem", lang="english", words_lower=True, no_punct_nodes=True, max_dist=4, max_weight=16, links_from_stopwords=True, links_to_stopwords=True):
    if normalization == "lem":
        words = words_lems(text, lower=words_lower)
    elif normalization == "stem":
        words = words_stems(text, lang=lang, lower=words_lower)

    words_map = words_to_int(words, ignore_punct=no_punct_nodes)

    return build_graph(words, words_map, max_dist=max_dist, max_weight=max_weight, links_from_stopwords=links_from_stopwords, links_to_stopwords=links_to_stopwords)
