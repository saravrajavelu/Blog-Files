import matplotlib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



def plotClusters(X, k, nearest_centroid, centroids, titleText = None):
    """
    Plots the K-Means graph
    """

    for i in range(k):
        cluster = X[nearest_centroid == i]
        plt.scatter(cluster[:, 0], cluster[:, 1])


    plt.scatter(centroids[:, 0], centroids[:, 1], marker = '*', color = 'black')

    if titleText:
        plt.title(titleText)
    plt.show()


def SSE(X, k, nearest_centroid, centroids):
    """
    Calculates the sum of squared error
    """
    sse = 0
    for i in range(k):
        sse += np.sum(np.sqrt(np.sum(np.power(X[nearest_centroid == i] - centroids[i], 2), axis = 1)))
    return sse


def KMeans(X, k, plot = False):
    """
    Input:
    X - 2D numpy array
    k - Number of clusters

    Process: Applies K-Means algorithm to the X data set with k clusters

    Output: (centroid_values, nearest_cluster)
            Tuple of centroid values and cluster membership details.
    """

    centroid_index = np.random.choice(X.shape[0], k)
    centroids = X[centroid_index, :]
    closest_centroid = lambda centroids, x: np.argmin(np.sqrt(np.sum(np.power(centroids -  x, 2), axis = 1)))

    counter = 0

    while True:

        old_centroids = centroids.copy()

        # Step 1: Finding nearest centroid for each point
        nearest_centroid = np.apply_along_axis(closest_centroid, 1, X, centroids)

        # Plots the graph every 2 iterations
        if plot:
            if counter % 2 == 0:
                plotClusters(X, k, nearest_centroid, centroids, 'Clusters after %d steps' % counter)
            counter += 1

        # Step 2: Updating centeroids
        for i in range(k):
            points = X[nearest_centroid == i]
            centroids[i] = np.mean(points, axis = 0) if points.shape[0] > 0 else centroids[i]


        # Convergence Condition
        if np.all(old_centroids == centroids):
            break

    if plot:
        plotClusters(X, k, nearest_centroid, centroids, 'Clusters after convergence')

    error = SSE(X, k, nearest_centroid, centroids)

    return centroids, nearest_centroid, error


# Setting up data
np.random.seed(123)
X = np.random.uniform(0, 10, (200, 2))

# Data points
plt.scatter(X[:, 0], X[:, 1], color = 'lightblue')
plt.title('Data Points')
plt.show()

# Algorithm
centroids, nearest_centroids, error = KMeans(X, 5, True)
