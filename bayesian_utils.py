import networkx as nx
import pandas as pd
import pgmpy
from matplotlib import pyplot as plt
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import BIFReader, BIFWriter
from sklearn.metrics import recall_score, precision_score

from data_utils import dataframe_choose_cols_and_sample


def bayesian_network_structure_learning(dataframe: pd.DataFrame, n_samples, discrete_bins, discrete_strategy):
    df = dataframe_choose_cols_and_sample(dataframe, n_samples, discrete_bins, discrete_strategy)
    net_estimator = HillClimbSearch(df)
    bayesian_network: pgmpy.base.DAG = net_estimator.estimate()
    bayesian_network_model = BayesianNetwork(bayesian_network.edges())
    bayesian_network_model.fit(df, estimator=MaximumLikelihoodEstimator, n_jobs=-1)
    #https://pgmpy.org/readwrite/bif.html è un tipo di formato per le reti bayesiane
    writer = BIFWriter(bayesian_network_model)
    writer.write_bif(filename=f'bayes_models\\bnet {n_samples} samples {discrete_bins} bins {discrete_strategy} strategy {len(df.columns)} features max likelihood.bif')
    G = nx.MultiDiGraph(bayesian_network_model.edges())
    pos = nx.spring_layout(G, iterations=100, k=2,
                           threshold=5, pos=nx.spiral_layout(G))
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color="#FFFF00")
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="normal",
        clip_on=True,
        horizontalalignment="center",
        verticalalignment="baseline",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=7,
        arrowstyle="->",
        edge_color="black",
        connectionstyle="arc3,rad=0.",
    )

    plt.title(f"Bayesian network")
    plt.savefig(f"bayes_models\\bayesian network {n_samples} samples {discrete_bins} bins {discrete_strategy} strategy {len(df.columns)} features.png")
    plt.show()
    plt.clf()


def bayesian_network_inference(model, data_to_predict):
    predicted_classes = []
    print("data", data_to_predict)
    inference = VariableElimination(model)
    amount = data_to_predict['Amount'].to_numpy()
    time = data_to_predict['Time'].to_numpy()
    v1 = data_to_predict['V1'].to_numpy()
    v2 = data_to_predict['V2'].to_numpy()
    v3 = data_to_predict['V3'].to_numpy()
    classes = data_to_predict['Class'].to_numpy()
    real_classes = [int(x) for x in classes]
    print("inizio a fare robe di inferenza")
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
    print(precision_score(real_classes, predicted_classes))
    print(recall_score(real_classes, predicted_classes))

def bayesian_network_simulate_samples(model:BayesianNetwork, n_samples):
    samples = model.simulate(n_samples=n_samples)
    print(samples)
    return samples

def get_bayesian_network_model(model_path):
    print("piglio il modello")
    reader = BIFReader(model_path)
    model = reader.get_model()
    print("modello pigliato")
    return model