#!/usr/bin/env python3

from inhs_outlining import *

from sklearn.model_selection import cross_val_score


def generate_dataset():
    np.random.seed(0)
    dataset = []
    for species in ['Lepomis Cyanellus', 'Notropis Stramineus', 'Gambusia Affinis', 'Phenacobius Mirabilis',
                    'Cyprinus Carpio', 'Esox Americanus']:
        all_from_species = Fish.query(select(Fish).where(Fish.genus + ' ' + Fish.species == species))
        np.random.shuffle(all_from_species)
        features = [list(fish.breen_features) + [species] for fish in all_from_species[:5]]
        dataset += features
    return np.array(dataset)


if __name__ == "__main__":
    with open("hyperparam_search_results.txt", 'w') as f:
        f.write("dark_thresh_mult,outline_connectivity,target_scale,feature_count,mean_5fold_crossval_acc\n")
        for dark_thresh_mult in np.arange(0.05, 3.05, 0.05):
            Fish.dark_thresh_mult = dark_thresh_mult
            for outline_connectivity in [4, 8]:
                Fish.outline_connectivity = outline_connectivity
                for target_scale in range(20, 125, 5):
                    Fish.spatial_resolution = target_scale
                    dataset = generate_dataset()
                    rows, cols = dataset.shape
                    maxfeats = cols - 1
                    X = dataset[:, :maxfeats].astype(float)
                    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
                    Y = dataset[:, maxfeats]
                    for feature_count in range(1, 41):
                        f.write(f"{dark_thresh_mult},{outline_connectivity},{target_scale},{feature_count}")
                        Fish.breen_feature_count = feature_count
                        current_xs = X[:, :feature_count]
                        clf = KNeighborsClassifier(n_neighbors=1)
                        accuracies = cross_val_score(clf, current_xs, Y)
                        accuracy = np.mean(accuracies)
                        f.write(f",{accuracy}\n")
