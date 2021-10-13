
import pickle
import re
from nltk.collocations import *
from nltk.stem.wordnet import WordNetLemmatizer
import gensim


def lemmatize(token):
    return WordNetLemmatizer().lemmatize(token, pos='v')

def tokenize(tweet):
    result = []
    for token in gensim.utils.simple_preprocess(tweet):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(lemmatize(token))
    res = ' '.join(result)
    return res

def preprocess_tweet(tweet):
    result = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    result = re.sub(r'(@[A-Za-z0-9-_]+)', '', result)
    result = re.sub(r'http\S+', '', result)
    result = re.sub(r'bit.ly/\S+', '', result) 
    result = re.sub(r'&[\S]+?;', '', result)
    result = re.sub(r'#', ' ', result)
    result = re.sub(r'[^\w\s]', r'', result)    
    result = re.sub(r'\w*\d\w*', r'', result)
    result = re.sub(r'\s\s+', ' ', result)
    result = re.sub(r'(\A\s+|\s+\Z)', '', result)
    processed = tokenize(result)
    return processed



def make_prediction(tweet):
    model = pickle.load(open("static/clf.pickle", "rb"))
    processed = preprocess_tweet(tweet)
    lst = []
    lst.append(processed)
    vec = pickle.load(open("static/vec.pickle", "rb"))
    vectorized = vec.transform(lst)
    pred = model.predict(vectorized)
    prob = model.predict_proba(vectorized)[:,1]
    mapping = {0: 'Same tweet, different day. Keep it movin\'.', 1: 'Didn\'t your parents ever wash your mouth out with SOAP?  Well they should!'}
    prediction = mapping[pred[0]]
    probability = str(prob)[1:-1]
    return tweet, prediction, probability