from inhs_outlining import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


def load_mat(mat_file):
    data = np.genfromtxt(mat_file, dtype=str, delimiter=',')
    np.random.seed(0)
    np.random.shuffle(data)
    # Data matrix CSVs contain a locus X&Y in the first two columns, but we ignore those in this function
    return data[:, 2:-1].astype(float), data[:, -1]


def drop_invariant_cols(X):
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


def pca_reduce(X, n_components=0.99):
    return PCA(random_state=0, n_components=n_components, whiten=False).fit_transform(X)


def make_top_k_scorer(k):
    return lambda clf, X, Y: top_k_accuracy_score(Y, clf.predict_proba(X), k=k)


f1_scorer = lambda clf, X, Y: f1_score(Y, clf.predict(X), average="weighted")
precision_scorer = lambda clf, X, Y: precision_score(Y, clf.predict(X), average="weighted")
recall_scorer = lambda clf, X, Y: recall_score(Y, clf.predict(X), average="weighted")


def run_cv_trials(clf, X, Y, folds=5, score=make_top_k_scorer(1)):
    print("Model:", clf)
    scores = cross_val_score(clf, X, Y, cv=folds, scoring=score)
    print("avg:   %.1f%%" % (scores.mean() * 100))
    print("std:   %.1f%%" % (scores.std() * 100))


def show_confusion_mat(clf, Xr, Y, folds=5):
    # Note that the predictions shown might not match what the classifier would predict in a cross_val_score() trial.
    y_pred = cross_val_predict(clf, Xr, Y, cv=folds)
    classes = np.unique(Y)
    cm = confusion_matrix(Y, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Greys", xticks_rotation=45)
    showplt()


def show_variation(fishes):
    encoding_mat = make_encoding_mat(fishes)
    ncols = encoding_mat.shape[1]
    pcs = pca_reduce(encoding_mat)
    efd_sets = [cross(fishes)]
    for i in range(3):
        col = pcs[:, i]
        efd_sets.append(encoding_mat[np.argmin(col)].reshape(ncols // 4, 4))
        efd_sets.append(encoding_mat[np.argmax(col)].reshape(ncols // 4, 4))
    colors = [
        (0xff, 0x0, 0x0),
        (0x7f, 0x0, 0x0),
        (0x0, 0xff, 0x0),
        (0x0, 0x7f, 0x0),
        (0x0, 0x0, 0xff),
        (0x0, 0x0, 0x7f),
        (0xff, 0xff, 0xff),
    ]
    show_contour(*[reconstruct(efds, 300, (0, 0)) for efds in efd_sets], colors=colors)


if __name__ == "__main__":
    salmos = Fish.all_of_genus("Salmo")
    show_variation(salmos)

