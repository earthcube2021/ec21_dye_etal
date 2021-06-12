import numpy as np


# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



#####        
# Plot a time series data set
#####
def plot_series(series, n_steps, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])
    
    

#####        
# Plot both actual data, with an overlay of predictions made by a model
#####  
def plot_multiple_forecasts(X, Y, Y_pred, n_steps):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0], n_steps)
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "ro-", label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "bx-", label="Forecast", markersize=10)
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)    
    
    
# def plot_clusters(X, y=None):
#     plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
#     plt.xlabel("$x_1$", fontsize=14)
#     plt.ylabel("$x_2$", fontsize=14, rotation=0)    

    
#####        
# Utility function for saving an image as a file external to a notebook
#####
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    # Where to save the figures
    PROJECT_ROOT_DIR = "."
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
    os.makedirs(IMAGES_PATH, exist_ok=True)
    
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)    
    

#####        
# Plot the centroids of a trained gaussian mixtures model
# Note that in this use case, there will only be once centroid - it's thr gradients we care about
#####    
def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)


    
#####        
# Plot the output of the gaussian mixtures processing
#####

def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 6, 96))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 6, 96),
                linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')
    
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)    
        

#####        
# Plot the overlay of the points flagged as outliers by the gaussian mixtures algortihm
#####    
def plot_gaussian_mixture_anomalies(gm, data_imputed, anomalies):
    plt.figure(figsize=(16, 10))

    plot_gaussian_mixture(gm, data_imputed)
    plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*')
    # plt.ylim(top=1.5, bottom= -0.5)
    
    

#####        
# For plotting of K-means graphs
#####
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


#####        
# Plot the centroids of the k-means regiois, laelling them with their numerical index
# this is important so a user can identify which regions should be retained as "clean" data, 
# and which regions should be removed 
#####    
def plot_decision_boundaries_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]

    
    #     Annotation article:
    #     https://nikkimarinsek.com/blog/7-ways-to-label-a-cluster-plot-python

    #     Print the index of the region at the middle of the decision boundary
    for i, centroid in enumerate(centroids):
        plt.annotate(i, 
                     centroid,
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=14, 
                     color=cross_color,
                     backgroundcolor=circle_color)     
    
#####        
# Plot the decision bountaries of the trained k-means algorithm
#####
def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_decision_boundaries_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)        
        
