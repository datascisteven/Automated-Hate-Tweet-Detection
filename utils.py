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
from config import *

def group_list(lst, size=100):
    """
    Generate batches of 100 ids in each
    Returns list of strings with , seperated ids
    """
    new_list =[]
    idx = 0
    while idx < len(lst):        
        new_list.append(
            ','.join([str(item) for item in lst[idx:idx+size]])
        )
        idx += size
    return new_list


def tweets_request(tweets_ids):
    """
    Make a requests to Tweeter API
    """
    df_lst = []
    
    for batch in tqdm(tweets_ids):
        url = "https://api.twitter.com/2/tweets?ids={}&tweet.fields=created_at&expansions=author_id&user.fields=created_at".format(batch)
        payload={}
        headers = {'Authorization': 'Bearer ' + config.bearer_token,
        'Cookie': 'personalization_id="v1_hzpv7qXpjB6CteyAHDWYQQ=="; guest_id=v1%3A161498381400435837'}
        r = requests.request("GET", url, headers=headers, data=payload)
        data = r.json()
        if 'data' in data.keys():
            df_lst.append(pd.DataFrame(data['data']))
    
    return pd.concat(df_lst)


def accuracy(y, y_hat):
    """
        Function to calculate accuracy score
        where y is original labels and y_hat is predicted labels
    """
    y_y_hat = list(zip(y, y_hat))
    tp = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 1])
    tn = sum([1 for i in y_y_hat if i[0] == 0 and i[1] == 0])
    return (tp + tn) / float(len(y_y_hat))

def f1(y, y_hat):
    """
        Function to calculate F1 score from precision and recall
        where y is original labels and y_hat is predicted labels
    """
    precision_score = precision(y, y_hat)
    recall_score = recall(y, y_hat)
    numerator = precision_score * recall_score
    denominator = precision_score + recall_score
    return 2 * (numerator / denominator)

def precision(y, y_hat):
    """
        Function to calculate precision score
        where y is original labels and y_hat is predicted labels
    """
    y_y_hat = list(zip(y, y_hat))
    tp = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 1])
    fp = sum([1 for i in y_y_hat if i[0] == 0 and i[1] == 1])
    return tp / float(tp + fp)

def recall(y, y_hat):
    """
        Function to calculate recall score
        where y is original labels and y_hat are predicted labels
    """
    y_y_hat = list(zip(y, y_hat))
    tp = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 1])
    fn = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 0])
    return tp / float(tp + fn)

def auc(X, y, model):
    """
        Function to calculate ROC AUC Score based on predict_proba(X)
        where X is feature values, y is target values, and model is instantiated model variable
    """
    probs = model.predict_proba(X)[:,1] 
    return roc_auc_score(y, probs)

def auc2(X, y, model):
    """
        Function to calculate ROC AUC Score based on decision_function(X)
        where X is feature values, y is target values, and model is instantiated model variable
    """
    probs = model.decision_function(X)
    return roc_auc_score(y, probs)

def aps(X, y, model):
    """
        Function to calculate PR AUC Score based on predict_proba(X)
        where X is feature values, y is target values, and model is instantiated model variable
    """
    probs = model.predict_proba(X)[:,1]
    return average_precision_score(y, probs)

def aps2(X, y, model):
    """
        Function to calculate PR AUC Score based on decision_function(X)
        where X is feature values, y is target values, and model is instantiated model variable
    """
    probs = model.decision_function(X)
    return average_precision_score(y, probs)

def get_metrics(X_tr, y_tr, X_val, y_val, y_pred_tr, y_pred_val, model):
    """
        Function to get training and validation accuracy, F1, ROC AUC, recall, precision, PR AUC scores
        Instantiate model and pass the model into function
        Pass X_train, y_train, X_val, Y_val datasets
        Pass in calculated model.predict(X) for y_pred
    """
    ac_tr = accuracy_score(y_tr, y_pred_tr)
    ac_val= accuracy_score(y_val, y_pred_val)
    f1_tr = f1_score(y_tr, y_pred_tr)
    f1_val = f1_score(y_val, y_pred_val)
    au_tr = auc(X_tr, y_tr, model)
    au_val = auc(X_val, y_val, model)
    rc_tr = recall_score(y_tr, y_pred_tr)
    rc_val = recall_score(y_val, y_pred_val)
    pr_tr = precision_score(y_tr, y_pred_tr)
    pr_val = precision_score(y_val, y_pred_val)
    aps_tr = aps(X_tr, y_tr, model)
    aps_val = aps(X_val, y_val, model)

    print('Training Accuracy: ', ac_tr)
    print('Validation Accuracy: ', ac_val)
    print('Training F1 Score: ', f1_tr)
    print('Validation F1 Score: ', f1_val)
    print('Training AUC Score: ', au_tr)
    print('Validation AUC Score: ', au_val)
    print('Training Recall Score: ', rc_tr)
    print('Validation Recall Score: ', rc_val)
    print('Training Precision Score: ', pr_tr)
    print('Validation Precision Score: ', pr_val)
    print('Training Average Precision Score: ', aps_tr)
    print('Validation Average Precision Score: ', aps_val)
    
    cnf = confusion_matrix(y_val, y_pred_val)
    group_names = ['TN','FP','FN','TP']
    group_counts = ['{0:0.0f}'.format(value) for value in cnf.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cnf.flatten()/np.sum(cnf)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cnf, annot=labels, fmt='', cmap='Blues', annot_kws={'size':16})

def num_of_words(df, col):
    df['word_ct'] = df[col].apply(lambda x: len(str(x).split(" ")))
    print(df[[col, 'word_ct']])

def num_of_chars(df, col):
    df['char_ct'] = df[col].str.len()
    print(df[[col, 'char_ct']])

def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

def avg_word_length(df, col):
    df['avg_wrd'] = df[col].apply(lambda x: avg_word(x))
    print(df[[col, 'avg_wrd']].head())

def hash_tags(df, col):
    df['hashtags'] = df[col].apply(lambda x: len(re.split(r'#', str(x)))-1)
    print(df[[col, 'hashtags']].head())

def preprocess_tweet(df, col):
    """
        Remove callouts, character references (HTML characters, emojis), # in hashtags, 
        Remove Twitter code RT and QT, URL links, punctuation, excess whitespace between
        Lowercase all words and remove leading and trailing whitespaces
    """
    df[col] = df[col].apply(lambda x: re.sub(r'@[\S]+', ' ', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'&[\S]+?;', ' ', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'#', ' ', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'(\bRT\b|\bQT\b)', ' ', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'http[\S]+', ' ', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', r'', str(x)))
    df[col] = df[col].apply(lambda x: " ".join(x.lower() for x in x.split()))
    df[col] = df[col].apply(lambda x: re.sub(r'\w*\d\w*', r' ', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'\s\s+', ' ', str(x)))

def tokenize(df, col):
    """
        Function to tokenize column of strings without punctuation
        Input into word_tokenize() must be string with spaces only
        Output is a list of tokenized words
    """
    text = ' '.join(df[col].to_list())
    tokens = nltk.word_tokenize(text)
    return tokens

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def no_stopwords(text):
    lst = [word for word in text if word not in stop_words]
    return lst

def term_frequency(df):
    tf1 = (df['tweet'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index())
    tf1.columns = ['words', 'tf']
    tf1 = tf1.sort_values(by='tf', ascending=False).reset_index()
    return tf1

def stemming(token_list):
    ss = PorterStemmer()
    lst = [ss.stem(w) for w in token_list]
    return lst

def lemmatization(df):
    df['lem'] = df['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return df['lem'].head()