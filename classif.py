from inhs_outlining import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import cross_val_score


def load_mat(mat_file):
    data = np.genfromtxt(mat_file, dtype=str, delimiter=',')
    np.random.seed(0)
    np.random.shuffle(data)
    return data[:, 2:-1].astype(float), data[:, -1]


def drop_invariant_cols(X, inds=None):
    inds = np.argwhere(np.all(X[..., :] == 0, axis=0))
    X = np.delete(X, inds, axis=1)
    return X, inds


def standardize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def normalize(X):
    Xn = []
    for efds in X:
        xni, trans = pyefd.normalize_efd(efds.reshape(len(efds) // 4, 4), return_transformation=True)
        Xn.append(xni.flatten()[3:] * (-1 if abs(trans[1]) > 0.25 else 1))
    return np.array(Xn)


def lda_reduce(X, Y):
    return LDA().fit_transform(X, Y)


def pca_reduce(X, Y, n_components=0.99):
    return PCA(random_state=0, n_components=n_components, whiten=False).fit_transform(X, Y)


def make_top_k_scorer(k):
    return lambda clf, X, Y: top_k_accuracy_score(Y, clf.predict_proba(X), k=k)


def run_top_k_cv_trials(clf, X, Y, folds=10, score=make_top_k_scorer(1)):
    print("Model:", clf)
    scores = cross_val_score(clf, X, Y, cv=folds, scoring=score)
    print("acc:   %.1f%%" % (scores.mean() * 100))
    print("std:   %.1f%%" % (scores.std() * 100))


if __name__ == "__main__":
    pass
