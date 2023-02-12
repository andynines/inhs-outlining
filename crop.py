import colorsys
import cv2.cv2 as cv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn import svm
import sys


def load_mask_for(inhs_id):
    # mask = cv.imread(f"{inhs_id}.png")
    mask = cv.imread("INHS_FISH_69099_mask.png")  # REMOVE ME
    _, mask = cv.threshold(mask, 0xfe, 0xff, cv.THRESH_BINARY)
    return mask


def crop(img, bbox):
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def show_set_distances(s1, s2, dist):
    all_point_pairs = itertools.product(s1, s2)
    all_point_distances = [dist(p1, p2) for p1, p2 in all_point_pairs]
    print("Average distance between all points:", np.mean(all_point_distances))

    s1_centroid = np.mean(s1)
    s2_centroid = np.mean(s2)
    print("Distance between set centroids:", dist(s1_centroid, s2_centroid))


def extract_unique_colors(img, mask, label):
    roi = img & mask
    colors = np.unique(roi.reshape(-1, roi.shape[-1]), axis=0)[1:]
    colors = np.concatenate([colors, np.repeat([label], len(colors), axis=0).reshape(-1, 1)], axis=1)
    return colors


def centroid_distance(points1, points2):
    return np.linalg.norm(np.mean(points1, axis=1) - np.mean(points2, axis=1))


def classify(data):
    np.random.shuffle(data)
    test = data[:len(data)//3]
    train = data[len(data)//3:]
    clf = svm.SVC(kernel="linear")
    clf.fit(train[:, :3], train[:, 3])
    predictions = clf.predict(test[:, :3])
    expecteds = test[:, 3]
    print("Accuracy:", np.sum(predictions == expecteds) / len(predictions))


if __name__ == "__main__":
    inhs_id = f"INHS_FISH_{sys.argv[1]}"
    img = cv.imread(f"some-ml-ready-images/{inhs_id}.jpg")
    with open("ml-ready-metadata.json", 'r') as f:  # Todo: speed this up using JSON streaming. See the ijson library.
        all_metadata = json.load(f)
        bbox = all_metadata[inhs_id]["fish"][0]["bbox"]
    img = crop(img, bbox)

    mask = load_mask_for(inhs_id)
    mask = crop(mask, bbox)
    foreground_colors = extract_unique_colors(img, mask, 1)
    background_colors = extract_unique_colors(img, ~mask, 0)
    all_color_data = np.unique(np.concatenate([foreground_colors, background_colors], axis=0), axis=0).astype(float)
    all_color_data[:, :3] /= 255.0
    fig = plt.figure()
    print("Foreground colors:", len(foreground_colors))
    print("Background colors:", len(background_colors))

    axis = fig.add_subplot(1, 2, 1, projection="3d")
    rs = all_color_data[:, 2]
    gs = all_color_data[:, 1]
    bs = all_color_data[:, 0]
    labels = all_color_data[:, 3]
    scatter_point_colors = ["green" if l else "black" for l in labels]
    axis.scatter(rs.flatten(), gs.flatten(), bs.flatten(), c=scatter_point_colors, marker='.')
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")

    axis = fig.add_subplot(1, 2, 2, projection="3d")
    hsvs = np.array([(*colorsys.rgb_to_hsv(r, g, b), label) for r, g, b, label in all_color_data], dtype=float)
    hs = hsvs[:, 0]
    ss = hsvs[:, 1]
    vs = hsvs[:, 2]
    axis.scatter(hs.flatten(), ss.flatten(), vs.flatten(), c=scatter_point_colors, marker='.')
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")

    plt.show()

    print("HSV space")
    classify(hsvs)

    print("RGB space")
    classify(all_color_data)
