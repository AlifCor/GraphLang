import nltk
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
    
def words_stems(text, lang="english", ignore_stopwords=False, ignore_punct=False):
    tokens = words_tokens(text, ignore_punct=ignore_punct)
    stemmer = SnowballStemmer(lang, ignore_stopwords=ignore_stopwords)
    
    return [stemmer.stem(t) for t in tokens]

def words_to_int(words, first_index=0, ignore_punct=False):
    if(ignore_punct):
        words_set = set([w for w in words if w not in string.punctuation])
    else:
        words_set = set(words)
    
    return {w:i for w,i in zip(words_set, range(first_index, first_index + len(words_set)))}
    