import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, auc, average_precision_score, confusion_matrix, roc_auc_score, plot_precision_recall_curve
from collections import Counter
from tqdm import tqdm
import config
# %reload_ext autoreload
# %autoreload 2
from utils import *


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
    y_y_hat = list(zip(y, y_hat))
    tp = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 1])
    tn = sum([1 for i in y_y_hat if i[0] == 0 and i[1] == 0])
    return (tp + tn) / float(len(y_y_hat))

def f1(y, y_hat):
    precision_score = precision(y, y_hat)
    recall_score = recall(y, y_hat)
    numerator = precision_score * recall_score
    denominator = precision_score + recall_score
    return 2 * (numerator / denominator)

def precision(y, y_hat):
    y_y_hat = list(zip(y, y_hat))
    tp = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 1])
    fp = sum([1 for i in y_y_hat if i[0] == 0 and i[1] == 1])
    return tp / float(tp + fp)

def recall(y, y_hat):
    # Your code here
    y_y_hat = list(zip(y, y_hat))
    tp = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 1])
    fn = sum([1 for i in y_y_hat if i[0] == 1 and i[1] == 0])
    return tp / float(tp + fn)

def auc(X, y, model):
    probs = model.predict_proba(X)[:,1] 
    return roc_auc_score(y, probs)

def auc2(X, y, model):
    probs = model.decision_function(X)
    return roc_auc_score(y, probs)

def aps(X, y, model):
    probs = model.predict_proba(X)[:,1]
    return average_precision_score(y, probs)

def aps2(X, y, model):
    probs = model.decision_function(X)
    return average_precision_score(y, probs)

def get_metric(X_tr, y_tr, X_val, y_val, y_pred_tr, y_pred_val, model):
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

def get_metrics(X_tr, y_tr, X_val, y_val, y_pred_tr, y_pred_val, model):
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

def get_metricz(X_tr, y_tr, X_val, y_val, y_pred_tr, y_pred_val, model):
    ac_tr = accuracy_score(y_tr, y_pred_tr)
    ac_val= accuracy_score(y_val, y_pred_val)
    f1_tr = f1_score(y_tr, y_pred_tr)
    f1_val = f1_score(y_val, y_pred_val)
    au_tr = auc2(X_tr, y_tr, model)
    au_val = auc2(X_val, y_val, model)
    rc_tr = recall_score(y_tr, y_pred_tr)
    rc_val = recall_score(y_val, y_pred_val)
    pr_tr = precision_score(y_tr, y_pred_tr)
    pr_val = precision_score(y_val, y_pred_val)
    aps_tr = aps2(X_tr, y_tr, model)
    aps_val = aps2(X_val, y_val, model)

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

def get_metrics_2(X_tr, y_tr, X_val, y_val, y_pred_tr, y_pred_val, model):
    ac_tr = accuracy_score(y_tr, y_pred_tr)
    ac_val= accuracy_score(y_val, y_pred_val)
    f1_tr = f1_score(y_tr, y_pred_tr)
    f1_val = f1_score(y_val, y_pred_val)
    au_tr = auc2(X_tr, y_tr, model)
    au_val = auc2(X_val, y_val, model)
    rc_tr = recall_score(y_tr, y_pred_tr)
    rc_val = recall_score(y_val, y_pred_val)
    pr_tr = precision_score(y_tr, y_pred_tr)
    pr_val = precision_score(y_val, y_pred_val)
    aps_tr = aps2(X_tr, y_tr, model)
    aps_val = aps2(X_val, y_val, model)

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

def plot_feature_importances(X, model):
    n_features = X.shape[1]
    plt.figure(figsize=(8, 8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns.values)
    plt.xlabel("Feature Importance")
    plt.ylabel('Feature')


def pr_curve(X, y, model):
    y_score = model.decision_function(X) 
    ap = average_precision_score(y, y_score)
    disp = plot_precision_recall_curve(model, X, y)
    disp.ax_.set_title('Precision-Recall Curve: AP={0:0.2f}'.format(ap))

def pr_curve2(X, y, model):
    y_score = model.predict_proba(X)[:,1]
    ap = average_precision_score(y, y_score)
    disp = plot_precision_recall_curve(model, X, y)
    disp.ax_.set_title('Precision-Recall Curve: AP={0:0.2f}'.format(ap))


def sampling(X_tr, y_tr, X_val, y_val, model, best_model):
    X_tr_res, y_tr_res = model.fit_resample(X_tr, y_tr)
    print("Training Count: ", Counter(y_tr_res))
    X_val_res, y_val_res = model.fit_resample(X_val, y_val)
    print("Validation Count: ", Counter(y_val_res))
    fit = best_model.fit(X_tr_res, y_tr_res)
    y_pred_tr = fit.predict(X_tr_res)
    y_pred_val = fit.predict(X_val_res)
    get_metric(X_tr_res, y_tr_res, X_val_res, y_val_res, y_pred_tr, y_pred_val, fit)
