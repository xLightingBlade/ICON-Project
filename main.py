import keras_tuner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


import data_utils
import neural_net_hypermodel
from bayesian_utils import bayesian_network_structure_learning, get_bayesian_network_model, \
    bayesian_network_simulate_samples, bayesian_network_inference

LABEL_GENUINE = 0
LABEL_FRAUD = 1
TEST_SIZE = .2
SCORING = ['accuracy', 'precision', 'recall', 'f1']
CV_FOLDS = 5

def main():
    og_dataset = data_utils.dataset_load("creditcard.csv", False)
    scaled_dataset = data_utils.dataset_scale(og_dataset)
    oversampled_dataset = data_utils.dataset_scale(og_dataset, True)
    data_utils.get_pie_chart_of_classes(oversampled_dataset, 'pics\\oversampled data pie chart.png')


    data_utils.correlation_matrix(scaled_dataset, 'pics\\scaled data correlations.png')
    data_utils.correlation_matrix(oversampled_dataset, 'pics\\oversampled data correlations.png')
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

    logreg_grid = {'C': [1e-1, 1e1, 1e3, 1e5]}
    rf_grid = {'n_estimators': [100, 150, 250], 'max_depth': [None, 5, 10]}
    knn_grid = {'n_neighbors': [2, 5, 10, 15]}
    svc_grid = {'C': [1e-1, 1e1, 1e3, 1e5]}
    dec_tree_grid = {'max_depth': [None, 5, 10]}
    grid_search_obj = [logreg_grid, rf_grid, knn_grid, svc_grid, dec_tree_grid]

    supervised_base_cross_validate(supervised_base_classifiers, x, y, False)
    supervised_base_cross_validate(supervised_base_classifiers, x_oversampled, y_oversampled, True)

    grid_search_hyperparam(supervised_base_classifiers, x, y, False, grid_search_obj)
    grid_search_hyperparam(supervised_base_classifiers, x_oversampled, y_oversampled, True, grid_search_obj)

    tsne_and_visualize(scaled_dataset, 80, 5000, 30000, oversampled=False)
    tsne_and_visualize(oversampled_dataset, 80, 5000, 50000, oversampled=True)


    #Isolation forest per visualizzare anomalie
    isolation_forest(scaled_dataset)
    isolation_forest(oversampled_dataset, True)

    #testo le varie combinazioni di discretizzazione su modelli base, per capire il giusto numero di bins
    #si lascia commentato poichè non più utile
    #data_discretization_test(scaled_dataset, supervised_base_classifiers)
    #results_df = pd.read_csv("csv\\discretization results.csv")
    #print(results_df.sort_values('recall', ascending=False))

    print('inizio bayes')
    #in realtà adesso il numero di feature usate per apprendere la rete è fisso a 5 + class, di più non si riesce
    bayesian_network_structure_learning(oversampled_dataset, 100000, 20, "kmeans")
    which_model = 'bayes_models\\bnet 100000 samples 20 bins kmeans strategy 6 features max likelihood.bif'
    bayesian_model = get_bayesian_network_model(which_model)
    generated_samples = bayesian_network_simulate_samples(bayesian_model, 10000)
    print(which_model)
    bayesian_network_inference(bayesian_model, generated_samples)
    #TODO: approccio fallimentare
    #samples = data_utils.dataframe_get_sample(oversampled_dataset, 100, 10, 'kmeans', to_discretize=True)
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
        results_df.to_csv('csv\\first cv no sampling.csv', index=False)
    else:
        results_df.to_csv('csv\\first cv with smote oversampling.csv', index=False)

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
    plt.savefig(f"pics\\tsne {'original scaled data' if not oversampled else 'oversampled data'} perp {perp} iters {iters}.png")
    plt.show()

def grid_search_hyperparam(classifiers, x, y, over_sampled, grids):
    i = 0
    results = []
    kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=False)
    for key, classifier in classifiers.items():
        grid = GridSearchCV(classifier, cv=kf, param_grid=grids[i], n_jobs=-1, verbose=True, refit='recall', scoring=SCORING, return_train_score=True)
        #grid sarà il best estimator restituito da gridsearchcv, che va refittato sull'intero dataset
        grid.fit(x, y)
        best_parameters = grid.best_params_
        best_scores_test = {
            "Accuracy(var)": f'{round(grid.cv_results_["mean_test_accuracy"][grid.best_index_], 5)}({round(grid.cv_results_["std_test_accuracy"][grid.best_index_] ** 2, 5)})',
            "Precision(var)": f'{round(grid.cv_results_["mean_test_precision"][grid.best_index_], 5)}({round(grid.cv_results_["std_test_precision"][grid.best_index_] ** 2, 5)})',
            "Recall(var)": f'{round(grid.cv_results_["mean_test_recall"][grid.best_index_], 5)}({round(grid.cv_results_["std_test_recall"][grid.best_index_] ** 2, 5)})',
            "F1(var)": f'{round(grid.cv_results_["mean_test_f1"][grid.best_index_], 5)}({round(grid.cv_results_["std_test_f1"][grid.best_index_] ** 2, 5)})',
        }
        best_scores_train = {
            "Accuracy(var)": f'{round(grid.cv_results_["mean_train_accuracy"][grid.best_index_], 5)}({round(grid.cv_results_["std_train_accuracy"][grid.best_index_] ** 2, 5)})',
            "Precision(var)": f'{round(grid.cv_results_["mean_train_precision"][grid.best_index_], 5)}({round(grid.cv_results_["std_train_precision"][grid.best_index_] ** 2, 5)})',
            "Recall(var)": f'{round(grid.cv_results_["mean_train_recall"][grid.best_index_], 5)}({round(grid.cv_results_["std_train_recall"][grid.best_index_] ** 2, 5)})',
            "F1(var)": f'{round(grid.cv_results_["mean_train_f1"][grid.best_index_], 5)}({round(grid.cv_results_["std_train_f1"][grid.best_index_] ** 2, 5)})',
        }
        results.append({"Classifier": key, "phase":"training", **best_parameters, **best_scores_train})
        results.append({"Classifier": key, "phase":"testing", **best_parameters, **best_scores_test})
        i = i+1
    results_df = pd.DataFrame(results)
    if over_sampled:
        path = 'csv\\grid search 3rd smote oversampling.csv'
        results_df.to_csv(path, index=False)
        data_utils.visualize_and_save_dataframe(results_df, path)
    else:
        path = 'csv\\grid search 3rd no sampling.csv'
        results_df.to_csv(path, index=False)
        data_utils.visualize_and_save_dataframe(results_df, path)

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
    results_df.to_csv("csv\\discretization results.csv", index=False)

def neural_net(dataframe:DataFrame, filename):
    inputs = dataframe.drop('Class', axis=1)
    target = dataframe['Class']
    x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model = neural_net_hypermodel.MyHypermodel(inputs)
    tuner = keras_tuner.RandomSearch(
        model,
        objective='val_loss',
        overwrite=True,
        max_trials=5)
    tuner.search(x_train, y_train, epochs = 30, validation_split = 0.2, callbacks= [stop_early])
    hp = tuner.get_best_hyperparameters()[0]
    hypermodel = tuner.hypermodel.build(hp)
    hypermodel.summary()
    history = hypermodel.fit(x_train, y_train, epochs=30, batch_size=32, validation_split = 0.2, callbacks= [stop_early])
    hypermodel.save(f'{filename}.keras')
    test_loss, test_acc, test_precision, test_recall = hypermodel.evaluate(x_test, y_test)
    f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    print("\nTest loss: {}, test accuracy: {}, test precision: {}, test recall: {}".
          format(test_loss, test_acc, test_precision, test_recall))
    print("F1 score: {}".format(f1))
    plot_neural_net(history, f'neural\\{filename} plot.png')

#presa in prestito dal mio scorso progetto
def plot_neural_net(history, file_name):
    fig, axs = plt.subplots(1, 2)

    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

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

    #Il mapping da -1,1 a 0,1 non è strettamente necessario però almeno siamo allineati a com'è il dataset
    dataframe["outlier"] = iso.predict(x)
    y_pred = dataframe["outlier"].map({1: 0, -1: 1})
    print(roc_auc_score(y, y_pred))
    print(recall_score(y, y_pred))
    dataframe["outlier_label"] = dataframe["outlier"].map({1: 0, -1: 1})
    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x)

    dataframe["PC1"] = x_pca[:, 0]
    dataframe["PC2"] = x_pca[:, 1]

    plt.figure(figsize=(10, 6))
    colors = {0: 'blue', 1: 'red'}
    plt.scatter(dataframe["PC1"], dataframe["PC2"],
                c=dataframe["outlier_label"].map(colors),
                alpha=0.6,
                s=10)
    plt.title("Isolation Forest plot")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    normal_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                               markersize=8, label='Normal')
    anomaly_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                markersize=8, label='Outlier')
    plt.legend(handles=[normal_dot, anomaly_dot])
    plt.savefig(f"pics\\isolation forest {'smote' if is_oversampled_dataset else ''}.png")
    plt.show()



if __name__ == "__main__":
    main()
