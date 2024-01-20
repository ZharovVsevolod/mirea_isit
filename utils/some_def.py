import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import classification_report
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import tree
import math

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import matplotlib.cm as cm

def compare_quality(y_predict, y_test, need_print_data=False):
    if need_print_data:
        print(y_predict)
        print("-----")
        print(np.array(y_test))
        print("-----")
    print(classification_report(np.array(y_test), y_predict))
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(np.array(y_test), y_predict, ax=ax, colorbar=False)
    plt.show()

# Функции для построения tsne и umap и быстрой их отрисовки
def some_tsne(data_sc, answers, perplex):
    T = TSNE(n_components=2, perplexity=perplex, random_state=567)
    TSNE_features = pd.DataFrame(data=T.fit_transform(data_sc), columns=["x", "y"])
    TSNE_features["keys"] = answers

    fig = plt.figure()
    sns.scatterplot(data=TSNE_features, x="x", y="y", hue="keys").set_title(f"TSNE {perplex}")
    return fig

def make_tsne(data_sc, answers, perplexities = [15, 25, 35]):
    figures = []
    for p in perplexities:
        figures.append(some_tsne(data_sc, answers, p))
    return figures

def make_umap(data_sc, answers, n_n = (5, 25, 50), m_d = (0.1, 0.6)):
    um = dict()
    for i in range(len(n_n)):
        for j in range(len(m_d)):
            um[(n_n[i], m_d[j])] = (umap.UMAP(n_neighbors=n_n[i], min_dist=m_d[j], random_state=567).fit_transform(data_sc))
            um[(n_n[i], m_d[j])] = pd.DataFrame(um[(n_n[i], m_d[j])], columns = ["x", "y"])
            um[(n_n[i], m_d[j])]["keys"] = answers
    return um

def plot_umap(um, n_n, m_d):
    figures = []
    for i in n_n:
        for j in m_d:
            fig = plt.figure()
            figures.append(sns.scatterplot(data=um[(i, j)], x="x", y="y", hue="keys").set_title(f"UMAP n_n={i}, m_d={j}"))
    return figures

def load_data():
    from ucimlrepo import fetch_ucirepo 
  
    # fetch dataset 
    ionosphere = fetch_ucirepo(id=52) 
    
    # data (as pandas dataframes) 
    X = ionosphere.data.features 
    y = ionosphere.data.targets

    return X, y

def make_svm(x_train, y_train):
    param_kernel = ["linear", "rbf", "poly", "sigmoid"]
    c_list = [3, 5, 6, 8]
    parameters = {"kernel" : param_kernel, "C" : c_list}
    model = SVC()
    grid_search_svm = GridSearchCV(estimator = model, param_grid = parameters)
    grid_search_svm.fit(x_train, y_train)

    print(grid_search_svm.best_score_)
    print(grid_search_svm.best_params_)

    return grid_search_svm.best_estimator_

def make_knn(x_train, y_train):
    number_of_neighbors = np.arange(2, 7)
    algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
    weights_need = ["uniform", "distance"]
    model_KNN = KNeighborsClassifier()
    params = {"n_neighbors" : number_of_neighbors, "algorithm" : algorithms, "weights" : weights_need}
    grid_search = GridSearchCV(estimator = model_KNN, param_grid = params)
    grid_search.fit(x_train, y_train)
    
    print(grid_search.best_score_)
    print(grid_search.best_params_)

    return grid_search.best_estimator_

def make_rf(x_train, y_train):
    n_estimators_need = np.arange(50, 150, 50)
    criterion_need = ["gini", "entropy", "log_loss"]
    max_depth_need = np.arange(8, 13)
    min_samples_split_need = np.arange(2, 4)

    model_RF = RF()
    params = {
        "n_estimators" : n_estimators_need, 
        "criterion" : criterion_need, 
        "max_depth" : max_depth_need, 
        "min_samples_split" : min_samples_split_need
    }
    grid_search = GridSearchCV(estimator = model_RF, param_grid = params)
    grid_search.fit(x_train, y_train)

    print(grid_search.best_score_)
    print(grid_search.best_params_)

    return grid_search.best_estimator_

def key(a):
    match a:
        case "g":
            return 0
        case "b":
            return 1

def count(s):
    count = {}
    for i in s:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    return count

def calculate_kn_distance(X, k, idx1=10, idx2=24):
    kn_distance = []
    for i in range(len(X)):
        eucl_dist = []
        for j in range(len(X)):
            eucl_dist.append(math.sqrt(
                (X[f"Attribute{idx1}"][i] - X[f"Attribute{idx1}"][j])**2 + 
                (X[f"Attribute{idx2}"][i] - X[f"Attribute{idx2}"][j])**2
            ))
        eucl_dist.sort()
        kn_distance.append(eucl_dist[k])
    return kn_distance

def sil_kl(X, range_n_clusters, idx1=10, idx2=24):
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.

        # clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
        clusterer = KMeans(
            n_clusters=n_clusters,
            init="k-means++", 
            max_iter=300, 
            n_init=10, 
            random_state=1702
        )
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[f"Attribute{idx1}"], X[f"Attribute{idx1}"],
            marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()