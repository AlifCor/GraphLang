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

def words_to_int(words, first_index=0, ignore_punct=False, ignore_stopwords=False, lang=None):
    if(ignore_stopwords and lang != None):
        stopwords = nltk.corpus.stopwords.words(lang)
        words = filter(lambda w: w not in stopwords, words)

    if(ignore_punct):
        words_set = set([w for w in words if w not in string.punctuation])
    else:
        words_set = set(words)

    return {w:i for w,i in zip(words_set, range(first_index, first_index + len(words_set)))}

def build_link(adj, weight, words_map, words, from_index, to_index, max_dist, stopwords, links_to_stopwords=True, self_links=False):
    words_len = len(words)
    links_made = 0
    while to_index < words_len and (words[to_index] in string.punctuation or (not links_to_stopwords and words[to_index] in stopwords) or (not self_links and words[to_index] == words[from_index])):
        to_index += 1
        weight /= 2

    if (to_index - from_index) <= max_dist and to_index < len(words):
        links_made = 1
        adj[words_map[words[from_index]], words_map[words[to_index]]] = adj[words_map[words[from_index]], words_map[words[to_index]]] + weight
    weight /= 2
    return weight, to_index + 1, links_made

def build_link2(adj, weight, words_map, words, from_index, to_index, max_dist, stopwords, links_to_stopwords=True, self_links=False):
    links_made = 0
    if(weight <= 0):
        weight, to_index + 1, links_made
    words_len = len(words)

    while to_index < words_len and (words[to_index] in string.punctuation or (not links_to_stopwords and words[to_index] in stopwords) or (not self_links and words[to_index] == words[from_index])):
        to_index += 1
        weight -= 1

    if (to_index - from_index) <= max_dist and to_index < len(words):
        links_made = 1
        adj[words_map[words[from_index]], words_map[words[to_index]]] = adj[words_map[words[from_index]], words_map[words[to_index]]] + weight
    weight -= 1
    return weight, to_index + 1, links_made

def build_graph(lemmas, lemmas_map, max_dist=20, nlinks=4, max_weight=16, lang=None, links_from_stopwords=True, links_to_stopwords=True, self_links=False):
    len_dist_lemmas = len(lemmas_map)
    len_lemmas = len(lemmas)
    adj = np.zeros((len_dist_lemmas, len_dist_lemmas))
    if(lang != None and (not links_from_stopwords or not links_to_stopwords)):
        stopwords = nltk.corpus.stopwords.words(lang)
    for index, lemma in enumerate(lemmas):
        # TODO Take into account stop words
        if lemma in string.punctuation or (not links_from_stopwords and lemma in stopwords):
            continue
        weight = max_dist#max_weight
        next_index = index + 1
        total_links_made = 0

        for i in range(0, max_dist):
            weight, next_index, links_made = build_link2(adj, weight, lemmas_map, lemmas, index, next_index, max_dist, stopwords, links_to_stopwords, self_links)
            total_links_made += links_made

            if(total_links_made >= nlinks or weight <= 0):
                break

    return adj

def text_to_graph(text, normalization="lem", lang="english", words_lower=True, no_punct_nodes=True, nlinks=4, max_dist=20, max_weight=16, ignore_stopwords=False, links_from_stopwords=True, links_to_stopwords=True, self_links=False, return_words_map=False):
    if(ignore_stopwords):
        links_from_stopwords = False
        links_to_stopwords = False

    if normalization == "lem":
        words = words_lems(text, lower=words_lower)
    elif normalization == "stem":
        words = words_stems(text, lang=lang, lower=words_lower)

    words_map = words_to_int(words, lang=lang, ignore_punct=no_punct_nodes, ignore_stopwords=ignore_stopwords)

    graph = build_graph(words, words_map, lang=lang, max_dist=max_dist, max_weight=max_weight, links_from_stopwords=links_from_stopwords, links_to_stopwords=links_to_stopwords)
    if(return_words_map):
        return (graph, words_map)
    else:
        return graph

def get_n_closest_words(graph, word_map, word, n_words=10):
    index = word_map[word]
    word_map_inversed = {i[1]:i[0] for i in word_map.items()}
    return [word_map_inversed[np.argsort(graph[index])[::-1][i]] for i in range(n_words)]
