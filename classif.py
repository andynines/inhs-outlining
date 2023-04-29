from inhs_outlining import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC


def load_mat(mat_file):
    data = np.genfromtxt(mat_file, dtype=str, delimiter=',')
    np.random.seed(0)
    np.random.shuffle(data)
    return data[:, 2:-1].astype(float), data[:, -1]


def standardize(X):
    zero_col_inds = np.argwhere(np.all(X[..., :] == 0, axis=0))
    X = np.delete(X, zero_col_inds, axis=1)
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def normalize(X):
    pass


def lda_reduce(X, Y):
    return LDA().fit_transform(X, Y)


def pca_reduce(X, Y, n_components=0.99):
    return PCA(random_state=0, n_components=n_components, whiten=False).fit_transform(X, Y)


def run_trial_with(clf, X, Y, top_ks=(1, 3, 5), folds=10):
    print("Model:", clf)
    for k in top_ks:
        scores = cross_val_score(clf, X, Y, cv=folds,
                                 scoring=lambda clf, X, Y: top_k_accuracy_score(Y, clf.predict_proba(X), k=k))
        print('-' * 5, "TOP", k, '-' * 5)
        print("acc:   %.1f%%" % (scores.mean() * 100))
        print("std:   %.1f%%" % (scores.std() * 100))


if __name__ == "__main__":
    X, Y = load_mat("1mm_fab_fifteen.csv")
    Xr = lda_reduce(X, Y)
    run_trial_with(
        SVC(random_state=0, kernel='linear', probability=True),
        Xr, Y,
    )
