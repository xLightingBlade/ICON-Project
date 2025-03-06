import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.lines as mlines
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from skopt.space import Real, Categorical, Integer
import json
import collections
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, \
    classification_report, make_scorer
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
LABEL_GENUINE = 0
LABEL_FRAUD = 1
TEST_SIZE = .2
SCORING = ['accuracy', 'precision', 'recall', 'f1']
CV_FOLDS = 5
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

def supervised_learning_cv(classifiers, x_train, y_train, over_sampled):
    if not over_sampled:
        print(f"before SMOTE sampling, counter(y) = {Counter(y_train)}")
        #normale cross validation stratificata (per dataset sbilanciati), con solo scaling di amount e time nel dataset
        with open("first_cv_results.txt", "w") as result_file:
            result_file.write(f'Prima normale crossvalidation, no oversampling. Classificatori : {list(classifiers.keys())}\n')
            for key, classifier in classifiers.items():
                classifier.fit(x_train, y_train)
                scores = cross_validate(classifier, x_train, y_train, n_jobs=-1, cv=CV_FOLDS, scoring=SCORING, verbose=1)
                result = (f"Classifiers: {classifier.__class__.__name__} has\n {scores['test_precision'].mean()} mean precision ({round(scores['test_precision'].std(), 5)} std)"
                    f"\n {scores['test_recall'].mean()} mean recall ({round(scores['test_recall'].std(), 5)} std)\n {scores['test_f1'].mean()} mean f1 score ({round(scores['test_f1'].std(), 5)} std)\n"
                    f"\n {scores['test_accuracy'].mean()} mean accuracy ({round(scores['test_accuracy'].std(), 5)} std)\n\n")
                result_file.write(result)
    else:
        # prova di oversampling sulla classe fraud (non undersampling: sulla perdita di informazioni ci si può fare poco ma sull`overfitting si può fare tanto)
        print(f"After SMOTE, counter(y) = {Counter(y_train)}")
        with open("smote_cv_results.txt", "w") as result_file:
            result_file.write(
                f'Crossvalidation dopo oversampling SMOTE. Classificatori : {list(classifiers.keys())}\n')
            for key, classifier in classifiers.items():
                classifier.fit(x_train, y_train)
                scores = cross_validate(classifier, x_train, y_train, n_jobs=-1, cv=CV_FOLDS, scoring=SCORING, verbose=1)
                result = (f"Classifiers: {classifier.__class__.__name__} has\n {scores['test_precision'].mean()} mean precision ({round(scores['test_precision'].std(), 5)} std)"
                    f"\n {scores['test_recall'].mean()} mean recall ({round(scores['test_recall'].std(), 5)} std)\n {scores['test_f1'].mean()} mean f1 score ({round(scores['test_f1'].std(), 5)} std)\n"
                    f"\n {scores['test_accuracy'].mean()} mean accuracy ({round(scores['test_accuracy'].std(), 5)} std)\n\n")
                result_file.write(result)

def hyperparameter_search_supervised_learning(x_train, y_train, x_test, y_test, over_sampled, bayes_search_objects):
    if not over_sampled:
        with open('bayes_search_no_smote.txt', 'w') as result_file:
            for opt in bayes_search_objects:
                opt.fit(x_train, y_train)
                result_file.write(f'Bayes Search, no oversampling. Classificatore: {opt.estimator}\n')
                result_file.write(f'{opt.score(x_test, y_test)}\n')
                result_file.write(f'{opt.best_score_}\n')
                result_file.write(f'{opt.best_estimator_}\n')
                result_file.write(f'{opt.best_params_}\n')
                result_file.write(f'{opt.cv_results_["params"][opt.best_index_]}\n')
    else:
        with open('bayes_search_with_smote.txt', 'w') as result_file:
            for opt in bayes_search_objects:
                opt.fit(x_train, y_train)
                result_file.write(f'Bayes Search, dati oversampling SMOTE. Classificatore: {opt.estimator}\n')
                result_file.write(f'{opt.score(x_test, y_test)}\n')
                result_file.write(f'{opt.best_score_}\n')
                result_file.write(f'{opt.best_estimator_}\n')
                result_file.write(f'{opt.best_params_}\n')
                result_file.write(f'{opt.cv_results_["params"][opt.best_index_]}\n')

def tsne_and_visualize(x_data, perp, iters):
    tsne = TSNE(n_components=2, perplexity=perp, learning_rate='auto', verbose=1, n_iter=iters)
    x_tsne = tsne.fit_transform(x_data)

    plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f'Plot T-SNE, perplexity {perp}, iterazioni {iters}')
    plt.savefig(f'tsne perp {perp} iters {iters}.png', bbox_inches='tight')
    plt.show()


def main():
    og_dataset = dataset_load_and_explore("creditcard.csv")
    scaled_dataset = dataset_scale(og_dataset)
    x = scaled_dataset.drop('Class', axis=1)
    y = scaled_dataset['Class']
    x_train_og, x_test_og, y_train_og, y_test_og = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    x_train = x_train_og.values
    x_test = x_test_og.values
    y_train = y_train_og.values
    y_test = y_test_og.values
    oversampler = SMOTE(sampling_strategy=0.5, random_state=42)
    x_train_oversampled, y_train_oversampled = oversampler.fit_resample(x_train, y_train)
    #TODO: raggruppare i vari dizionari sotto uno unico, del tipo che logistic regression diviene
    #a sua volta un dizionario con modello e griglia di parametri
    supervised_base_classifiers = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "KNearest": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(),
        "DecisionTreeClassifier": DecisionTreeClassifier()
    }
    log_param_search = BayesSearchCV(
        supervised_base_classifiers["LogisticRegression"],
        {
            'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
        },
        cv=CV_FOLDS,
        scoring=SCORING,
        refit='f1',
        verbose=2
    )
    svc_param_search = BayesSearchCV(
        supervised_base_classifiers["Support Vector Classifier"],
        {
            'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
        },
        cv=CV_FOLDS,
        scoring=SCORING,
        refit='f1',
        verbose=2
    )
    
    rf_param_search = BayesSearchCV(
        supervised_base_classifiers["RandomForest"],
        {
            'n_estimators': [75,150,200],
            'max_depth': [2,5,8],
        },
        cv=CV_FOLDS,
        scoring=SCORING,
        refit='f1',
        verbose=2
    )
    
    knn_param_search = BayesSearchCV(
        supervised_base_classifiers["KNearest"],
        {
            'n_neighbors': [2,5,10,20],
        },
        cv=CV_FOLDS,
        scoring=SCORING,
        refit='f1',
        verbose=2
    )
    
    dec_tree_param_search = BayesSearchCV(
        supervised_base_classifiers["DecisionTreeClassifier"],
        {
            'max_depth': [2, 5, 7, 10],
        },
        cv=CV_FOLDS,
        scoring=SCORING,
        refit='f1',
        verbose=2
    )    
    #search_obj = [log_param_search, rf_param_search, svc_param_search, knn_param_search, dec_tree_param_search]
    #supervised_learning_cv(supervised_base_classifiers, x_train, y_train, False)
    #supervised_learning_cv(supervised_base_classifiers, x_train_oversampled, y_train_oversampled, True)
    #hyperparameter_search_supervised_learning(x_train, y_train, x_test, y_test, False, search_obj)
    #hyperparameter_search_supervised_learning(x_train_oversampled, y_train_oversampled, x_test, y_test, True, search_obj)
    #tsne_and_visualize(x_train_oversampled, 50, 3000)
    #tsne_and_visualize(x_train_oversampled, 100, 3000)
    #tsne_and_visualize(x_train_oversampled, 100, 10000)
    # https://www.kaggle.com/code/sifodhara/credit-card-fraud-detection-using-isolation-forest per dei plot sui dati, da mettere qua per abbellire sto progetto
    # Initialize and fit the Isolation Forest model
    contamination_ratio = scaled_dataset["Class"].sum() / len(scaled_dataset)
    print("Contamination ratio:", contamination_ratio)

    iso = IsolationForest(contamination=contamination_ratio, n_estimators=200, random_state=42, n_jobs=-1, verbose=2)
    iso.fit(x)

    # Predict anomalies: 1 = inlier, -1 = outlier
    scaled_dataset["anomaly"] = iso.predict(x)
    # Map the predictions: 1 -> 0 (normal) and -1 -> 1 (anomaly)
    scaled_dataset["anomaly_label"] = scaled_dataset["anomaly"].map({1: 0, -1: 1})
    # Reduce data to 2 principal components for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(x)

    # Add PCA components to the dataframe for plotting
    scaled_dataset["PC1"] = X_pca[:, 0]
    scaled_dataset["PC2"] = X_pca[:, 1]

    # Create a scatter plot with different colors for normal and anomalous points
    plt.figure(figsize=(10, 6))
    colors = {0: 'blue', 1: 'red'}
    plt.scatter(scaled_dataset["PC1"], scaled_dataset["PC2"],
                c=scaled_dataset["anomaly_label"].map(colors),
                alpha=0.6,
                s=10)
    plt.title("Isolation Forest: Anomaly Detection on Credit Card Fraud Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # Create custom legend
    normal_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                               markersize=8, label='Normal')
    anomaly_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                markersize=8, label='Anomaly')
    plt.legend(handles=[normal_dot, anomaly_dot])
    plt.savefig("isolation forest.png")
    plt.show()



if __name__ == "__main__":
    main()
