import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, auc, average_precision_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
import re
import nltk
from nltk.stem.porter import PorterStemmer
from textblob import Word
import pandas as pd
import requests
import sys
sys.path.append("../py")
from config import keys
import gensim
from nltk.stem import WordNetLemmatizer


def remove_users(df, col):
    """
    Function to remove usernames in retweets and callouts
    df: name of dataframe
    col: name of column containing Twitter text
    """
    df[col] = df[col].apply(lambda x: re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', str(x))) # remove re-tweet
    df[col] = df[col].apply(lambda x: re.sub(r'(@[A-Za-z0-9-_]+)', '', str(x))) # remove tweeted at

def remove_charef(df, col):
    """
    Function to remove character references
    df: name of dataframe
    col: name of column containing Twitter text   
    """
    df[col] = df[col].apply(lambda x: re.sub(r'&[\S]+?;', '', str(x)))


def remove_hashtags(df, col):
    """
    Function to remove the hash from hashtags
    df: name of dataframe
    col: name of column containing Twitter text     
    """
    df[col] = df[col].apply(lambda x: re.sub(r'#', ' ', str(x)))

def remove_av_qt(df, col):
    """Takes a column of strings in Pandas dataframe and removes AUDIO/VIDEO tags or labels"""    
    df[col] = df[col].apply(lambda x: re.sub(r'(\bQT\b)', '', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'VIDEO:', '', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'AUDIO:', '', str(x)))

def remove_links(df, col):
    # df['links'] = df[col].apply(lambda x: re.findall(r'http\S+', str(x))) 
    df[col] = df[col].apply(lambda x: re.sub(r'http\S+', '', str(x)))  # remove http links
    # df['links'] = df[col].apply(lambda x: re.findall(r'bit.ly/\S+', str(x))) 
    df[col] = df[col].apply(lambda x: re.sub(r'bit.ly/\S+', '', str(x)))  # remove bit.ly links    
    # df.links = df.links.apply(lambda x: str(x)[1:-1]) # remove brackets around list

def remove_punctuation(df, col):    
    df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', r'', str(x)))

def lowercasing(df, col):
    df[col] = df[col].apply(lambda x: " ".join(x.lower() for x in x.split()))

def remove_numerics(df, col):
    """function to remove numbers or words with digits"""
    df[col] = df[col].apply(lambda x: re.sub(r'\w*\d\w*', r'', str(x)))

def remove_whitespaces(df, col):
    """function to remove any double or more whitespaces to single and any leading and trailing whitespaces"""
    df[col] = df[col].apply(lambda x: re.sub(r'\s\s+', ' ', str(x))) 
    df[col] = df[col].apply(lambda x: re.sub(r'(\A\s+|\s+\Z)', '', str(x))) 

def lemmatize(token):
    """Returns lemmatization of a token"""
    return WordNetLemmatizer().lemmatize(token, pos='v')

def tokenize(tweet):
    """Returns tokenized representation of words in lemma form excluding stopwords"""
    result = []
    for token in gensim.utils.simple_preprocess(tweet):
        if token not in gensim.parsing.preprocessing.STOPWORDS \
                and len(token) > 2:  # drops words with less than 3 characters
            result.append(lemmatize(token))
    return result

def tokenize_and_lemmatize(df, col):
    df[col] = df[col].apply(lambda x: tokenize(x))

def preprocess_tweets(df, col):
    """master function to preprocess tweets"""
    collect_and_remove_users(df, col)
    collect_and_remove_charef(df, col)
    collect_and_remove_hashtags(df, col)
    remove_links(df, col)
    remove_av_qt(df, col)
    remove_punctuation(df, col)
    lowercasing(df, col)
    remove_whitespaces(df, col)
    remove_numerics(df, col)
    tokenize_and_lemmatize(df, col)