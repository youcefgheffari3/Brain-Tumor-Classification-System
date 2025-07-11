from sklearn.decomposition import PCA

def apply_pca(X, n_components=0.95):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca
