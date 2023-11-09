#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#2.3


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 50 samples, 2 features
X_small = pd.read_csv("C:/Users/Patron/Desktop/data/data2D.csv",names=np.arange(0,2).tolist()).to_numpy()

# 500 samples, 1000 features
X_large = pd.read_csv("C:/Users/Patron/Desktop/data/data1000D.csv",names=np.arange(0,1000).tolist()).to_numpy() 


# In[4]:


def plot(X_orig, X_reconstructed, title):
    plt.scatter(X_orig[:,0],X_orig[:,1],marker='o',facecolors='none', edgecolors='b',s=20,label="Original Datapoints")
    plt.scatter(X_reconstructed[:,0],X_reconstructed[:,1], marker='x',c='r',s=20,label="Reconstructed Datapoints")
    plt.xticks(np.arange(0, 11, 1))
    plt.yticks(np.arange(0, 11, 1))
    plt.legend()
    plt.title(title,fontweight="bold");


# In[5]:


def squared_sum_of_differences(X_orig, X_reconstructed):
    return np.sum(np.square(X_orig - X_reconstructed))/len(X_orig)


# In[6]:


#buggyPCA
def buggy_pca(X, d):
    """
    param X: m x n input matrix with n-dimensions
    param d: desired dimension of reduced matrix
    
    return: tuple (eigenvalues, reconstructed matrix)
    """
    
    # store input matrix dimensions
    m,n = X.shape
    
    # SVD of X
    U, s, V = np.linalg.svd(X)
    eigenvalues = s[0:d]**2
    
    # low dimensional representation of X
    Z = np.dot(X,V[0:d].T)
    
    # reduced version of X
    X_r = np.dot(Z.reshape(m,d),V[0:d].reshape(d,n))
    
    return (eigenvalues, X_r)


# In[7]:


eigenvalues, X_r = buggy_pca(X=X_small, d=1)
buggy_pca_ssd = squared_sum_of_differences(X_small, X_r)
print("Squared Sum of Differences: %.6f"%buggy_pca_ssd)


# In[8]:


plot(X_small, X_r, "Buggy PCA")


# In[10]:


#Demeaned PCA
def demeaned_pca(X, d):
    """
    param X: m x n input matrix with n-dimensions
    param d: desired dimension of reduced matrix
    
    return: tuple (eigenvalues, reconstructed matrix)
    """
    
    # store input matrix dimensions
    m,n = X.shape

    # subtract each dimension by its corresponding mean
    X_mean = X - X.mean(axis=0)
    
    
    # SVD of X
    U, s, V = np.linalg.svd(X_mean)
    eigenvalues = s[0:d]**2
    
    # low dimensional representation of X
    Z = np.dot(X_mean,V[0:d].T)
    
    # reduced version of X
    X_r = np.dot(Z.reshape(m,d),V[0:d].reshape(d,n)) + X.mean(axis=0)
    
    return (eigenvalues, X_r)


# In[11]:


eigenvalues, X_r = demeaned_pca(X=X_small, d=1)
demeaned_pca_ssd = squared_sum_of_differences(X_small, X_r)
print("Squared Sum of Differences: %.6f"%demeaned_pca_ssd)


# In[12]:


plot(X_small, X_r, "Demeaned PCA")


# In[14]:


#NormalizedPCA
def normalized_pca(X, d):
    """
    param X: m x n input matrix with n-dimensions
    param d: desired dimension of reduced matrix
    
    return: tuple (eigenvalues, reconstructed matrix)
    """
    
    # store input matrix dimensions
    m,n = X.shape

    # normalized the data
    X_mean = (X - X.mean(axis=0))
    X_sigma = np.sqrt(np.sum(np.square(X_mean),axis=0)/len(X))
    X_norm = X_mean / X_sigma
    
    # SVD of X
    U, s, V = np.linalg.svd(X_norm)
    eigenvalues = s[0:d]**2
    
    # low dimensional representation of X
    Z = np.dot(X_norm,V[0:d].T)
    
    # reduced version of X
    X_r = (np.dot(Z.reshape(m,d),V[0:d].reshape(d,n))*X_sigma) + X.mean(axis=0)
    
    return (eigenvalues, X_r)


# In[15]:


eigenvalues, X_r = normalized_pca(X=X_small, d=1)
normalized_pca_ssd = squared_sum_of_differences(X_small, X_r)
print("Squared Sum of Differences: %.6f"%normalized_pca_ssd)


# In[16]:


plot(X_small, X_r, "Normalized PCA")


# In[18]:


#DRO
def DRO(X, d):
    """
    param X: m x n input matrix with n-dimensions
    param d: desired dimension of reduced matrix
    
    return: tuple (eigenvalues, reconstructed matrix)
    """
    
    # store input matrix dimensions
    m,n = X.shape

    # eigen decomp
    X_mean = (X - X.mean(axis=0))
    eigenvalues, eigenvectors = np.linalg.eig(np.cov(X_mean.T))
    
    # sort eigenvector & eigenvalue pairs
    sorted_indices = np.argsort(-np.abs(eigenvalues))
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = -1*eigenvectors[:, sorted_indices]

    # low dimensional representation of X
    Z = np.dot(X_mean,eigenvectors.T[0:d].T)
    
    # reduced version of X
    X_r = np.dot(Z.reshape(m,d),eigenvectors.T[0:d].reshape(d,n)) + X.mean(axis=0)
    
    return (eigenvalues, eigenvectors, X_r)


# In[19]:


(eigenvalues, eigenvectors, X_r) = DRO(X=X_small, d=1)
dro_ssd = squared_sum_of_differences(X_small, X_r)
print("Squared Sum of Differences: %.6f"%dro_ssd)


# In[20]:


plot(X_small, X_r, "DRO")


# In[21]:


# dro clear knee point
X_mean = (X_large - X_large.mean(axis=0))
eigenvalues, eigenvectors = np.linalg.eig(np.cov(X_mean.T))
s = np.nan_to_num(np.sqrt(eigenvalues.real))
sum_singular_values = sum(s)
singular_value_proportions = []
for s_i in s:
    singular_value_proportions.append(s_i / sum_singular_values)
cumulative_singular_value_proportions = np.cumsum(singular_value_proportions)

# plotting
plt.figure(figsize=(12, 8))
plt.xticks(np.arange(0,1025,50))
plt.yticks(np.arange(0,101,10))
plt.grid()
plt.ylabel("% Variance Explained",fontsize=15)
plt.xlabel("Number of Features",fontsize=15)
plt.plot(cumulative_singular_value_proportions*100);


# In[22]:


(eigenvalues, X_r) = buggy_pca(X=X_large, d=500)
buggy_pca_ssd_1k = squared_sum_of_differences(X_large, X_r)
print("Squared Sum of Differences:",buggy_pca_ssd_1k)


# In[23]:


(eigenvalues, X_r) = demeaned_pca(X=X_large, d=500)
demeaned_pca_ssd_1k = squared_sum_of_differences(X_large, X_r)
print("Squared Sum of Differences:",demeaned_pca_ssd_1k)


# In[24]:


(eigenvalues, X_r) = normalized_pca(X=X_large, d=500)
normalized_pca_ssd_1k = squared_sum_of_differences(X_large, X_r)
print("Squared Sum of Differences:",normalized_pca_ssd_1k)


# In[25]:


(eigenvalues, eigenvectors, X_r) = DRO(X=X_large, d=500)
dro_ssd_1k = squared_sum_of_differences(X_large, X_r.real)
print("Squared Sum of Differences:",dro_ssd_1k)


# In[26]:


print("Reconstruction Errors 2-d Dataset:")
print("\tBuggy PCA: %.8f" % buggy_pca_ssd)
print("\tDemeaned PCA: %.8f" % demeaned_pca_ssd)
print("\tNormalized PCA: %.8f" % normalized_pca_ssd)
print("\tDRO: %.8f" % dro_ssd)

print("\nReconstruction Errors 1000-d Dataset:")
print("\tBuggy PCA:", buggy_pca_ssd_1k)
print("\tDemeaned PCA:", demeaned_pca_ssd_1k)
print("\tNormalized PCA:", normalized_pca_ssd_1k)
print("\tDRO:", dro_ssd_1k)


# In[ ]:




