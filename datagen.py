from inhs_outlining import *


def datamat(fishes):
    efds = [fish.efds.ravel() for fish in fishes]
    maxcoeffs = max(len(row) for row in efds)
    efds = [np.pad(row, (0, maxcoeffs - len(row)), "constant", constant_values=0) for row in efds]
    mat = np.array(efds)
    labels = np.array([f"{fish.genus} {fish.species}" for fish in fishes]).reshape(-1, 1)
    return np.concatenate((mat, labels), axis=1)


if __name__ == "__main__":
    data = datamat(Fish.all())
    np.savetxt("all.csv", data, fmt='%s', delimiter=',')
