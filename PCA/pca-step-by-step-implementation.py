import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def generate_synthetic_data():
    """
        This function generate synthetic data for our experimentation. 

        Output: A dataframe that contains 'Height' and 'Age' of a set of people.
    """
    data = {
        'Height': [170, 165, 180, 175, 160, 172, 168, 177, 162, 158, 181, 185, 173],
        'Age': [65, 59, 75, 68, 55, 70, 62, 74, 58, 54, 23, 32, 45]
    }

    df = pd.DataFrame(data)

    return df

def standardization(df):
    """
        This function standardizes our data to ensure that each feature has the same scale and contributes equally in the model.

        Input: df
        Output: X_standardized = (X_original - Mean(Xi)/standard deviation(Xi))
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    return df_scaled

def define_matrix_A(df):
    """
        This function converts our dataframe into a matrix.

        Input: The standardized data.
        Output: The corresponding matrix.
    """
    A = []

    for i in range(df.shape[0]):
        A.append(np.array(df[i]))

    return np.array(A)

def compute_covariance_matrix(A):
    """
        This function computes the covariance matrix of the scaled data.

        Input: The scaled data.
        Output: The corresponding covariance matrix. 
    """
    Cx = np.cov(A, rowvar=False)

    return Cx

def eigen_decomposition(Cx):
    """
        This function computes the eigenvectors decomposition of the covariance matrix Cx and defines the projection matrix W.

        Input: The covariance matrix Cx.
        Output: The projection matrix W, the list of the eigenvectors and eigenvalues.

        To reduce the dimensionality, you need to decomment the line break to ensure that you take only one eigenvector since we have only two.
    """
    W = []

    eigenvalues, eigenvectors = np.linalg.eig(Cx)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    for i in range(eigenvectors.shape[0]):
        W.append(np.array(eigenvectors[i]))
        #break

    return np.array(W), eigenvectors, eigenvalues

def projected_points(W,X):
    """
        This function converts each original point X into their corresponding point Y.

        Input: The scaled data, and the projection matrix.
        Output: The principal components Y.
    """
    Y = np.dot(W,X)

    return Y


def visualize(df,df2,eigvecs):
    """
        This function display:
        
        the original data df, 
        the projected data df2,
        the eigenvectors eigvecs.

        Note that you need to replace np.zeros_like(df2) by df2[:, 1] if your projected data remain in a 2-dimensional space. 
        Or, you can comment the first part of code to use the second part, which is currently comment but, the original and 
        the projected data points are in the same plot in this case.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(df[:,0], df[:,1], s=20, alpha=0.5, c='blue', edgecolors = 'none', label="Original point")
    axes[0].set_title("Original points")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    origin = [0, 0]
    for i in range(2):
        axes[0].quiver(
            origin[0], origin[1],
            eigvecs[0, i], eigvecs[1, i],
            angles='xy', scale_units='xy', scale=1
        )
    
    axes[1].scatter(df2[:,0], df2[:, 1], s=20, alpha=0.5, c='red', edgecolors='none', label="Projected points")
    axes[1].set_title("Projected points")
    axes[1].set_xlabel("X")
    axes[1].set_yticks([])
    
    plt.tight_layout()
    plt.show()
    """
    fig = plt.figure(figsize=(4,4))

    plt.scatter(df[:, 0], df[:, 1], s=10, alpha=0.5, c='red', edgecolors='none', label="Original points")

    origin = [0, 0]
    for i in range(2):
        plt.quiver(
            origin[0], origin[1],
            eigvecs[0, i], eigvecs[1, i],
            angles='xy', scale_units='xy', scale=1
        )

    plt.scatter(df2[:, 0], df2[:, 1], s=10, alpha=0.5, c='blue', edgecolors='none', label="Original points")
    plt.title("2D clouds of points")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.show()"""

def pca(X_scaled):
    pca = PCA(n_components=2)
    X_proj = pca.fit_transform(X_scaled)

    return pca, X_proj
    


if __name__ == "__main__":
    # Generate our test data
    df = generate_synthetic_data()

    # Apply min max standardozation to ensure that our data have the same scale
    df_scaled = standardization(df)
    
    # Define the Nxd matrix (here N=10 and d=2)
    A = define_matrix_A(df_scaled)
    
    # Compute the covariance matrix Cx
    Cx = compute_covariance_matrix(A)

    # Apply eigenvector decomposition to the covariance matrix A and define the projection matrix W
    W, eigvecs, eigvals = eigen_decomposition(Cx)
    
    # Project the data on the new directions and print the explained variance on each new direction.
    Y = projected_points(A,W)
    #print(np.var(y) for y in Y)
    
    #visualize the two cloud of points
    visualize(df_scaled, Y, eigvecs)
    
    # Let's verify whether the covariance matrix of Y is a diagonal matrix
    if W.shape[0] >= 2:
        Cy = compute_covariance_matrix(Y)
        print(W)
        print(eigvals)
        print(Cy)

    #Compare our code with the PCA implementation from sklearn
    pca, X_proj = pca(df_scaled)
    visualize(df_scaled, X_proj, eigvecs)
    print("Explained variance:", pca.explained_variance_)
    
