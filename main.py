from collections import Counter

import keras_tuner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas
import pandas as pd
import pgmpy.base
import seaborn as sns
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from pandas import DataFrame
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import BIFWriter, BIFReader
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, roc_curve, roc_auc_score)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV

import data_utils
import neural_net_hypermodel

LABEL_GENUINE = 0
LABEL_FRAUD = 1
TEST_SIZE = .2
SCORING = ['accuracy', 'precision', 'recall', 'f1']
CV_FOLDS = 5

def main():
    og_dataset = data_utils.dataset_load("creditcard.csv", False)
    scaled_dataset = data_utils.dataset_scale(og_dataset)
    oversampled_dataset = data_utils.dataset_scale(og_dataset, True)
    #data_utils.correlation_matrix(scaled_dataset, 'scaled data correlations.png')
    #data_utils.correlation_matrix(oversampled_dataset, 'oversampled data correlations.png')
    x = scaled_dataset.drop('Class', axis=1)
    y = scaled_dataset['Class']
    x_oversampled = oversampled_dataset.drop('Class', axis=1)
    y_oversampled = oversampled_dataset['Class']
    x_train, x_test, y_train, y_test = data_utils.split_x_and_y(x, y)
    x_train_oversampled, x_test_oversampled, y_train_oversampled, y_test_oversampled = data_utils.split_x_and_y(x_oversampled, y_oversampled)

    supervised_base_classifiers = {
        "LogisticRegression": LogisticRegression(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "KNearest": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(random_state=42),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42)
    }

    logreg_grid = {'C': [1e-3, 1e-1, 1e1, 1e3, 1e5]}
    rf_grid = {'n_estimators': [100, 150, 250], 'max_depth': [None, 5, 8],}
    knn_grid = {'n_neighbors': [2, 5, 10, 15]}
    svc_grid = {'C': [1e-3, 1e-1, 1e1, 1e3, 1e5]}
    dec_tree_grid = {'max_depth': [None, 5, 10, 15]}
    grid_search_obj = [logreg_grid, rf_grid, knn_grid, svc_grid, dec_tree_grid]

    #supervised_base_cross_validate(supervised_base_classifiers, x, y, False)
    #supervised_base_cross_validate(supervised_base_classifiers, x_oversampled, y_oversampled, True)

    #grid_search_hyperparam(supervised_base_classifiers, x_train, y_train, x_test, y_test, False, grid_search_obj)
    #grid_search_hyperparam(supervised_base_classifiers, x_train_oversampled, y_train_oversampled, x_test_oversampled, y_test_oversampled, True, grid_search_obj)

    #tsne_and_visualize(scaled_dataset, 80, 5000, 30000, oversampled=False)
    #tsne_and_visualize(oversampled_dataset, 80, 5000, 50000, oversampled=True)


    #Isolation forest per visualizzare anomalie
    isolation_forest(scaled_dataset)
    isolation_forest(oversampled_dataset, True)

    #testo le varie combinazioni di discretizzazione su modelli base, per capire il giusto numero di bins
    #data_discretization_test(scaled_dataset, supervised_base_classifiers)
    #results_df = pd.read_csv("discretization results.csv")
    #print(results_df.sort_values('f1', ascending=False))

    #print('inizio bayes')
    #bayesian_network_structure_learning(scaled_dataset, 1000, 20, "kmeans")
    #which_model = 'bnet 100000 samples 20 bins kmeans strategy 6 features max likelihood.bif'
    #bayesian_model = get_bayesian_network_model(which_model)
    #generated_samples = bayesian_network_simulate_samples(bayesian_model, 10000) #accuracy 0.9544
    #bayesian_network_inference(bayesian_model, generated_samples)
    #samples = dataframe_get_sample(oversampled_dataset, 10000, True)
    #bayesian_network_inference(bayesian_model, samples)
    #neural_net(oversampled_dataset, 'neural net')


def supervised_base_cross_validate(classifiers, x, y, over_sampled):
    results_df = pd.DataFrame()
    if not over_sampled:
        print(f"before SMOTE sampling, counter(y) = {Counter(y)}")
    else:
        print(f"After SMOTE, counter(y) = {Counter(y)}")
    #normale cross validation stratificata (per dataset sbilanciati), con solo scaling di amount e time nel dataset
    for key, classifier in classifiers.items():
        kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=False)
        scores = cross_validate(classifier, x, y, n_jobs=-1, cv=kf, scoring=SCORING, verbose=1)
        row = {"Classifier": key}
        for metric in SCORING:
            row[f"{metric}_mean"] = np.mean(scores[f"test_{metric}"])
            row[f"{metric}_std"] = np.std(scores[f"test_{metric}"])
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    if not over_sampled:
        results_df.to_csv('first cv no sampling.csv', index=False)
    else:
        results_df.to_csv('first cv with smote oversampling.csv', index=False)


def tsne_and_visualize(dataframe, perp, iters, n_samples, oversampled=False):
    if oversampled:
        df2 = pd.concat([dataframe[dataframe.Class == 1].sample(n_samples // 5), dataframe[dataframe.Class == 0].sample(n_samples)], axis=0)
    else:
        df2 = pd.concat([dataframe[dataframe.Class == 1], dataframe[dataframe.Class == 0].sample(n_samples)],axis=0)
    y = df2.iloc[:,-1]
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, learning_rate='auto', verbose=1, n_iter=iters, n_jobs=-1)
    x_transformed = tsne.fit_transform(df2)
    color_map = {0: 'blue', 1: 'red'}
    plt.figure()
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x_transformed[y == cl, 0],
                    y=x_transformed[y == cl, 1],
                    c=color_map[idx],
                    label=cl)
    plt.xlabel('X in t-SNE')
    plt.ylabel('Y in t-SNE')
    plt.legend(loc='upper left')
    plt.title('t-SNE visualization, samples from dataset')
    plt.savefig(f"tsne {'original scaled data' if not oversampled else 'oversampled data'} perp {perp} iters {iters}.png")
    plt.show()

def grid_search_hyperparam(classifiers, x_train, y_train, x_test, y_test, over_sampled, grids):
    i = 0
    results = []
    kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=False)
    for key, classifier in classifiers.items():
        grid = GridSearchCV(classifier, cv=kf, param_grid=grids[i], n_jobs=-1, verbose=True, refit='recall', scoring=SCORING)
        grid.fit(x_train, y_train)
        best_parameters = grid.best_params_
        best_scores = {
            "Best Accuracy": grid.cv_results_["mean_test_accuracy"][grid.best_index_],
            "Best Precision": grid.cv_results_["mean_test_precision"][grid.best_index_],
            "Best Recall": grid.cv_results_["mean_test_recall"][grid.best_index_],
            "Best F1": grid.cv_results_["mean_test_f1"][grid.best_index_],
        }
        results.append({"Classifier": key, **best_parameters, **best_scores})
        i = i+1
    results_df = pd.DataFrame(results)
    if over_sampled:
        results_df.to_csv('grid search smote oversampling.csv', index=False)
    else:
        results_df.to_csv('grid search no sampling.csv', index=False)


def dataframe_get_sample(dataframe:DataFrame, n_samples, to_discretize=False):
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    fraud_df = dataframe.loc[dataframe['Class'] == 1].sample(n_samples // 2)
    real_df = dataframe.loc[dataframe['Class'] == 0].sample(n_samples // 2)
    samples = pd.concat([fraud_df, real_df], ignore_index=True)
    if to_discretize:
        discrete_sample = discretize_df(samples, 20, 'kmeans')
        return discrete_sample
    return samples

def data_discretization_test(data, classifiers):
    x = data.drop('Class', axis=1)
    y = data['Class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)

    bins_list = [5, 10, 15, 20, 30, 50]
    strategies = ["uniform", "quantile", "kmeans"]

    results = []

    for strategy in strategies:
        for bins in bins_list:
            for classifier in classifiers.values():
                kbins = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
                x_train_discrete = kbins.fit_transform(x_train)
                x_test_discrete = kbins.transform(x_test)

                classifier.fit(x_train_discrete, y_train)

                y_pred = classifier.predict(x_test_discrete)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                results.append({
                    "classifier": classifier.__class__.__name__,
                    "strategy": strategy,
                    "n_bins": bins,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                })
                print(
                    f"Classifier: {classifier.__class__.__name__}, Strategy: {strategy}, n_bins: {bins} | F1: {f1:.4f},"
                    f" Precision: {precision:.4f}, Recall: {recall:.4f}\n")

    results_df = pd.DataFrame(results)
    print("\nResults:")
    print(results_df)
    results_df.to_csv("discretization results.csv", index=False)

def discretize_df(dataframe:DataFrame, bins, strategy):
    print('prima')
    print(dataframe.head())
    kbins = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
    cols = [col for col in dataframe.columns if col != 'Class']
    x_discrete = kbins.fit_transform(dataframe[cols])
    discrete_df = pd.DataFrame(x_discrete, columns=cols, index=dataframe.index)
    discrete_df['Class'] = dataframe['Class']
    discrete_df = discrete_df[dataframe.columns]
    print(discrete_df.head())
    return discrete_df

#tipizzo il parametro dataframe per aiutarmi con l`IDE
def bayesian_network_structure_learning(dataframe: pandas.DataFrame, n_samples, discrete_bins, discrete_strategy):
    df = dataframe_choose_cols_and_sample(dataframe, n_samples, discrete_bins, discrete_strategy)
    net_estimator = HillClimbSearch(df)
    bayesian_network: pgmpy.base.DAG = net_estimator.estimate()
    bayesian_network_model = BayesianNetwork(bayesian_network.edges())
    bayesian_network_model.fit(df, estimator=MaximumLikelihoodEstimator, n_jobs=-1)
    #https://pgmpy.org/readwrite/bif.html è un tipo di formato per le reti bayesiane
    writer = BIFWriter(bayesian_network_model)
    writer.write_bif(filename=f'bnet {n_samples} samples {discrete_bins} bins {discrete_strategy} strategy {len(df.columns)} features max likelihood.bif')
    #TODO: modificare un pò la visualizzazione della rete, troppo obvious cosi
    G = nx.MultiDiGraph(bayesian_network_model.edges())
    pos = nx.spring_layout(G, iterations=100, k=2,
                           threshold=5, pos=nx.spiral_layout(G))
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#ff574c")
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
        clip_on=True,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=7,
        arrowstyle="->",
        edge_color="purple",
        connectionstyle="angle3,angleA=90,angleB=0",
        min_source_margin=1,
        min_target_margin=1,
        edge_vmin=2,
        edge_vmax=2,
    )

    plt.title("BAYESIAN NETWORK GRAPH")
    plt.savefig(f"bayesian network {n_samples} samples {discrete_bins} bins {discrete_strategy} strategy {len(df.columns)} features.png")
    plt.show()
    plt.clf()

def dataframe_choose_cols_and_sample(dataframe:DataFrame, n_samples, bins, strategy):
    #prima uno shuffle del dataframe
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    x = dataframe[['Time', 'V1', 'V2', 'V3', 'Amount']]
    y = dataframe['Class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    sm = SMOTE(random_state=42, sampling_strategy=.5)
    x_train_oversampled, y_train_oversampled = sm.fit_resample(x_train, y_train)
    x_train = pd.DataFrame(x_train_oversampled, columns=x_train.columns)
    y_train = pd.Series(y_train_oversampled)
    x_train['Class'] = y_train
    fraud_df = x_train.loc[x_train['Class'] == 1 ].sample(n_samples // 2)
    real_df = x_train.loc[x_train['Class'] == 0].sample(n_samples // 2)
    df_sample = pd.concat([fraud_df, real_df], ignore_index=True)
    discrete_sample = discretize_df(df_sample, bins, strategy)
    print("discrete sample df")
    print(discrete_sample.head())
    return discrete_sample

def bayesian_network_inference(model, data_to_predict):
    predicted_classes = []
    inference = VariableElimination(model)
    amount = data_to_predict['Amount'].to_numpy()
    time = data_to_predict['Time'].to_numpy()
    v1 = data_to_predict['V1'].to_numpy()
    v2 = data_to_predict['V2'].to_numpy()
    v3 = data_to_predict['V3'].to_numpy()
    classes = data_to_predict['Class'].to_numpy()
    real_classes = [int(x) for x in classes]
    for i in range(len(data_to_predict)):
        data = {
            "Amount" : amount[i],
            "V1": v1[i],
            "V2": v2[i],
            "V3": v3[i],
            "Time": time[i]
        }
        res = inference.query(variables = ['Class'], evidence = data)
        class_res = inference.map_query(variables = ['Class'], evidence = data)
        #print(res)
        #print(class_res)
        predicted_classes.append(int(class_res['Class']))
    print(accuracy_score(real_classes, predicted_classes))

def bayesian_network_simulate_samples(model:BayesianNetwork, n_samples):
    samples = model.simulate(n_samples=n_samples)
    print(samples)
    return samples

def get_bayesian_network_model(model_path):
    reader = BIFReader(model_path)
    model = reader.get_model()
    return model

def neural_net(dataframe:DataFrame, filename):
    inputs = dataframe.drop('Class', axis=1)
    target = dataframe['Class']
    x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model = neural_net_hypermodel.MyHypermodel(inputs)
    tuner = keras_tuner.RandomSearch(
        model,
        objective='val_loss',
        overwrite=True,
        max_trials=5)
    tuner.search(x_train, y_train, epochs = 100, validation_split = 0.2, callbacks= [stop_early])
    hp = tuner.get_best_hyperparameters()[0]
    hypermodel = tuner.hypermodel.build(hp)
    hypermodel.summary()
    history = hypermodel.fit(x_train, y_train, epochs=100, batch_size=32, validation_split = 0.2, callbacks= [stop_early])
    hypermodel.save(f'{filename}.keras')
    hypermodel.evaluate(x_test, y_test)
    plot_neural_net(history, f'{filename} plot.png')

def plot_neural_net(history, file_name):
    fig, axs = plt.subplots(1, 2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    text = f"Learning rate 0.0001, 100 epochs."
    plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(file_name)
    plt.show()

def isolation_forest(dataframe, is_oversampled_dataset = False):
    print(dataframe.info())
    contamination = dataframe["Class"].sum() / len(dataframe)
    x = dataframe.drop('Class', axis=1)
    y = dataframe['Class']
    iso = IsolationForest(contamination=contamination, n_estimators=200, random_state=42, n_jobs=-1, verbose=2)
    iso.fit(x)

    # Predict anomalies: 1 = inlier, -1 = outlier
    dataframe["outlier"] = iso.predict(x)
    y_pred = dataframe["outlier"].map({1: 0, -1: 1})
    print(roc_auc_score(y, y_pred))
    print(recall_score(y, y_pred))
    # Map the predictions: 1 -> 0 (normal) and -1 -> 1 (anomaly)

    dataframe["outlier_label"] = dataframe["outlier"].map({1: 0, -1: 1})
    # Reduce data to 2 principal components for visualization
    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x)

    # Add PCA components to the dataframe for plotting
    dataframe["PC1"] = x_pca[:, 0]
    dataframe["PC2"] = x_pca[:, 1]

    # Create a scatter plot with different colors for normal and anomalous points
    plt.figure(figsize=(10, 6))
    colors = {0: 'blue', 1: 'red'}
    plt.scatter(dataframe["PC1"], dataframe["PC2"],
                c=dataframe["outlier_label"].map(colors),
                alpha=0.6,
                s=10)
    plt.title("Isolation Forest plot")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # Create custom legend
    normal_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                               markersize=8, label='Normal')
    anomaly_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                markersize=8, label='Outlier')
    plt.legend(handles=[normal_dot, anomaly_dot])
    plt.savefig(f"isolation forest {'smote' if is_oversampled_dataset else ''}.png")
    plt.show()



if __name__ == "__main__":
    main()
