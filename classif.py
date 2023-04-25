from inhs_outlining import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

data = np.genfromtxt("selected.csv", dtype=str, delimiter=',')
np.random.shuffle(data)
X = data[:, :-1].astype(float)
Y = data[:, -1]

# zero_col_inds = np.argwhere(np.all(X[..., :] == 0, axis=0))
# X = np.delete(X, zero_col_inds, axis=1)
# X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

pca = PCA(n_components=0.95, whiten=True) # 95 is best? Need whiten?
pca.fit(X)
Xr = X @ pca.components_.T

clf = KNeighborsClassifier(n_neighbors=5)
#clf = SVC()
scores = cross_val_score(clf, X, Y, cv=10)
print("accuracy:", scores.mean())
print("std dev:", scores.std())
