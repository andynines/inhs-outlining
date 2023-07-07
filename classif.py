from inhs_outlining import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
        xni, trans = pyefd.normalize_efd(efds.reshape(-1, 4), return_transformation=True)
        Xn.append(xni.flatten()[3:] * (-1 if abs(trans[1]) > 0.25 else 1))
    return np.array(Xn)


def synthesize_n_rows_from(real_rows, n):
    synth_rows = []
    while n > 0:
        num_refs = np.random.randint(2, real_rows.shape[0] + 1)
        np.random.shuffle(real_rows)
        refs = real_rows[:num_refs, :]
        weights = np.random.dirichlet(np.ones(num_refs), size=1)
        synth_rows.append(np.sum(refs * weights.T, axis=0))
        n -= 1
    return np.array(synth_rows)


def lda_reduce(X, Y):
    return LDA().fit_transform(X, Y)


def pca_reduce(X, n_components=0.99):
    return PCA(random_state=0, n_components=n_components, whiten=False).fit_transform(X)


def make_top_k_scorer(k):
    return lambda clf, X, Y: top_k_accuracy_score(Y, clf.predict_proba(X), k=k)


f1_scorer = lambda clf, X, Y: f1_score(Y, clf.predict(X), average="weighted")
precision_scorer = lambda clf, X, Y: precision_score(Y, clf.predict(X), average="weighted")
recall_scorer = lambda clf, X, Y: recall_score(Y, clf.predict(X), average="weighted")
pipeline = lambda clf: Pipeline([
    ("standardize", StandardScaler()),
    ("reduce", LDA()),
    ("classify", clf)
])


def run_cv_trials(clf, X, Y, folds=5, score=make_top_k_scorer(1)):
    print("Model:", clf)
    np.random.seed(0)
    #scores = cross_val_score(pipeline(clf), X, Y, scoring=score, cv=folds)
    scores = []
    kf = KFold(n_splits=folds)
    for (train_inds, test_inds) in kf.split(X):
        Xtrain, Ytrain = X[train_inds], Y[train_inds]
        Xmean = np.mean(Xtrain, axis=0)
        Xstd = np.std(Xtrain, axis=0)
        Xstd[Xstd == 0] = 1
        Xtrain = (Xtrain - Xmean) / Xstd
        lda = LDA()
        Xtrain = lda.fit_transform(Xtrain, Ytrain)
        clf.fit(Xtrain, Ytrain)
        Xtest = ((X[test_inds] - Xmean) / Xstd) @ lda.scalings_
        scores.append(score(clf, Xtest, Y[test_inds]))
    scores = np.array(scores)
    print("avg:   %.1f%%" % (scores.mean() * 100))
    print("std:   %.1f%%" % (scores.std() * 100))


def show_confusion_mat(clf, X, Y, folds=5):
    # Note that the predictions shown might not match what the classifier would predict in a cross_val_score() trial.
    y_pred = cross_val_predict(pipeline(clf), X, Y, cv=folds)
    classes = np.unique(Y)
    cm = confusion_matrix(Y, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Greys", xticks_rotation=45)
    showplt()


def show_variation_better(fishes):
    n = max(len(fish.normalized_outline) for fish in fishes)
    all_efds = np.array(pad_ragged([list(fish.encoding[0].ravel()) for fish in fishes]))
    avg = np.mean(all_efds, axis=0)
    std = np.sqrt(((all_efds - avg) ** 2).sum(axis=0) / len(fishes))
    recons = lambda efds: reconstruct(efds.reshape(-1, 4), n, (0, 0))
    contours = [recons(efds) for efds in [avg, avg + std, avg - std]]
    show_contour(*contours, colors=[(0xff, 0, 0), (0, 0, 0xff), (0xff, 0xff, 0xff)])


if __name__ == "__main__":
    pass
