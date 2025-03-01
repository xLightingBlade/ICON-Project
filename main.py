import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import json
import collections
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
LABEL_GENUINE = 0
LABEL_FRAUD = 1
TEST_SIZE = .2
SCORING = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
#TODO : LE PARAMETER GRID PER I CLASSIFICATORI

def dataset_load_and_explore(path):
    og_dataset = pd.read_csv(path)
    print(og_dataset.head())
    print(og_dataset[pd.isnull(og_dataset).any(axis=1)])   #nessuna riga con valori NaN
    print(og_dataset.dtypes)
    for column in og_dataset.columns:
       print(f'Column : {column} \n Variance: {og_dataset[column].var()}\n')

    genuine_proportion = round(og_dataset["Class"].value_counts(normalize=True)[0] * 100, 2)
    fraud_proportion = round(100 - genuine_proportion, 2)
    print(fraud_proportion)
    print(genuine_proportion)
    return og_dataset

def dataset_scale(data):
    scaler = RobustScaler()
    columns_to_scale = data[['Amount', 'Time']]
    columns_to_scale = columns_to_scale.to_numpy()
    scaled = scaler.fit_transform(columns_to_scale)
    scaled_columns = pd.DataFrame(scaled, columns=['Amount', 'Time'])
    print(scaled_columns)
    scaled_dataset = data.copy()
    scaled_dataset['Amount'] = scaled_columns['Amount']
    scaled_dataset['Time'] = scaled_columns['Time']
    return scaled_dataset

def supervised_learning(classifiers, x_train, y_train, over_sampled, scoring=None):
    if scoring is None:
        scoring = SCORING
    if not over_sampled:
        print(f"before SMOTE sampling, counter(y) = {Counter(y_train)}")
        #normale cross validation stratificata (per dataset sbilanciati), con solo scaling di amount e time nel dataset
        with open("first_results.txt", "w") as result_file:
            result_file.write(f'Prima normale classificazione, no oversampling. Classificatori : {list(classifiers.keys())}\n')
            for key, classifier in classifiers.items():
                classifier.fit(x_train, y_train)
                scores = cross_validate(classifier, x_train, y_train, n_jobs=-1, cv=10, scoring=scoring, verbose=1)
                result = (f"Classifiers: {classifier.__class__.__name__} has\n {scores['test_precision'].mean()} mean precision (+-{round(scores['test_precision'].std() * 2, 5)})"
                        f"\n {scores['test_recall'].mean()} mean recall (+-{round(scores['test_recall'].std() * 2, 5)})\n {scores['test_f1'].mean()} mean f1 score (+-{round(scores['test_f1'].std() * 2, 5)})"
                        f"\n {scores['test_roc_auc'].mean()} mean AUROC (+-{round(scores['test_roc_auc'].std() * 2, 5)})\n"
                        f"\n {scores['test_accuracy'].mean()} mean accuracy (+-{round(scores['test_accuracy'].std() * 2, 5)}\n\n")
                result_file.write(result)
    else:
        # prova di oversampling sulla classe fraud (non undersampling: sulla perdita di informazioni ci si può fare poco ma sull`overfitting si può fare tanto)
        oversampler = SMOTE(sampling_strategy=0.5, random_state=42)
        x_train, y_train = oversampler.fit_resample(x_train, y_train)
        print(f"After SMOTE, counter(y) = {Counter(y_train)}")
        with open("smote_results.txt", "w") as result_file:
            result_file.write(
                f'Classificazione dopo oversampling SMOTE. Classificatori : {list(classifiers.keys())}\n')
            for key, classifier in classifiers.items():
                classifier.fit(x_train, y_train)
                scores = cross_validate(classifier, x_train, y_train, n_jobs=-1, cv=10, scoring=scoring, verbose=1)
                result = (
                    f"Classifiers: {classifier.__class__.__name__} has\n {scores['test_precision'].mean()} mean precision (+-{round(scores['test_precision'].std() * 2, 5)})"
                    f"\n {scores['test_recall'].mean()} mean recall (+-{round(scores['test_recall'].std() * 2, 5)})\n {scores['test_f1'].mean()} mean f1 score (+-{round(scores['test_f1'].std() * 2, 5)})"
                    f"\n {scores['test_roc_auc'].mean()} mean AUROC (+-{round(scores['test_roc_auc'].std() * 2, 5)})\n"
                    f"")
                result_file.write(result)

def hyperparameter_search_supervised_learning(classifiers, x_train, y_train, over_sampled, scoring=None):
    if scoring is None:
        scoring = SCORING
    if not over_sampled:
        with open("hyperparameter_tuning_no_smote.txt", "w") as result_file:
            result_file.write(
                f'Classificazione con hyperparameter tuning, no SMOTE. Classificatori : {list(classifiers.keys())}\n')
            for key, classifier in classifiers.items():
                continue
                #TODO: LA GRID SEARCH


def main():
    og_dataset = dataset_load_and_explore("creditcard.csv")
    scaled_dataset = dataset_scale(og_dataset)
    x = scaled_dataset.drop('Class', axis=1)
    y = scaled_dataset['Class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values
    supervised_base_classifiers = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "KNearest": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(),
        "DecisionTreeClassifier": DecisionTreeClassifier()
    }
    supervised_learning(supervised_base_classifiers, x_train, y_train, False)
    supervised_learning(supervised_base_classifiers, x_train, y_train, True)


if __name__ == "__main__":
    main()