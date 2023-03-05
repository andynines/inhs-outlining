from pathlib import Path
import socket
from sqlalchemy import LargeBinary, String, Float, create_engine, select
from sqlalchemy.sql import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from werkzeug.utils import cached_property
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from typing import List, Tuple


def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.rad2deg(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))


def extract_k_most_significant_frequencies(signal, k):
    n = len(signal)
    dft = list(zip(np.fft.fft(signal), n * np.fft.fftfreq(n)))
    highest_term_freqs = sorted(dft[1:n // 2], key=lambda p: -p[0])[:k]
    return np.round(np.array(highest_term_freqs)[:, 1].real)


def make_dataset(genus, species):
    genus_fish = Fish.query(select(Fish).where((Fish.genus == genus) & (Fish.species == species)))
    xs = np.array([fish.features for fish in genus_fish])
    ys = np.array([f"{genus} {species}"] * len(xs)).reshape(-1, 1)
    return np.concatenate((xs, ys), axis=1)


def assert_is_lab_server():
    assert socket.gethostname() == "CCI-DX4M513", "Not on lab server"


class Base(DeclarativeBase):
    pass


class Fish(Base):
    __tablename__ = "fish"

    engine = create_engine("sqlite:///fish.db")

    avg_scale = None
    feature_count = 10

    @classmethod
    def sesh(cls, callback):
        with Session(cls.engine) as session:
            return callback(session)

    @classmethod
    def query(cls, stmt):
        return cls.sesh(lambda s: s.scalars(stmt).all())

    @classmethod
    def with_id(cls, fid: str):
        return cls.query(select(cls).where(cls.id == fid))[0]

    @classmethod
    def all(cls):
        return cls.query(select(cls))

    @classmethod
    def calc_avg_scale(cls):
        if cls.avg_scale is None:
            cls.avg_scale = cls.query(select(func.avg(cls.scale)))[0]
        return cls.avg_scale

    @classmethod
    def count_fish_per_genus(cls):
        return dict(cls.sesh(lambda s: s.query(cls.genus, func.count(cls.genus)).group_by(cls.genus).all()))

    @classmethod
    def count_unique_species(cls):
        counts = cls.sesh(
            lambda s: s.query(cls.genus, cls.species, func.count(cls.id)).group_by(cls.genus, cls.species).all())
        counts.sort(key=lambda count: -count[2])
        return {f"{count[0]} {count[1]}": count[2] for count in counts}

    @classmethod
    def example_of(cls, genus, species):
        return cls.query(select(cls).where((cls.genus == genus) & (cls.species == species)))[0]

    # IDs aren't purely numeric! Some have underscores in them.
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    genus: Mapped[str] = mapped_column(String(50))
    species: Mapped[str] = mapped_column(String(50))
    image: Mapped[bytes] = mapped_column(LargeBinary)  # = cv.imencode('.jpg', img)[1].tobytes(),
    side: Mapped[str] = mapped_column(String(10))
    scale: Mapped[float] = mapped_column(Float)

    def __repr__(self) -> str:
        return f"<INHS_FISH_{self.id}>"

    @cached_property
    def cropped_im(self) -> np.array:
        """
        img[
          max(0, bbox[1] - BBOX_PAD_PX): bbox[3] + BBOX_PAD_PX,
          max(0, bbox[0] - BBOX_PAD_PX): bbox[2] + BBOX_PAD_PX,
        ]
        """
        nparr = np.frombuffer(self.image, np.uint8)
        im = cv.imdecode(nparr, cv.IMREAD_COLOR)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)

    @cached_property
    def original_im(self) -> np.array:
        assert_is_lab_server()
        path_template = f"/usr/local/bgnn/inhs_{{group}}/INHS_FISH_{self.id}.jpg"
        validation_path = Path(path_template.format(group="validation"))
        if validation_path.exists():
            im = cv.imread(validation_path)
        else:
            im = cv.imread(Path(path_template.format(group="test")))
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)

    @cached_property
    def mask(self) -> np.array:
        hsvim = cv.cvtColor(self.cropped_im, cv.COLOR_RGB2HSV)
        satim = hsvim[:, :, 1]
        # plt.hist(satim.ravel(), 256, [0,256])
        otsu_thresh, _ = cv.threshold(satim, 0, 0xff, cv.THRESH_BINARY | cv.THRESH_OTSU)
        dark_px = satim[satim < otsu_thresh].flatten()
        dark_mean = np.mean(dark_px)
        dark_std = np.std(dark_px)
        sat_std_bias_factor = 0.25
        new_thresh = dark_mean + sat_std_bias_factor * dark_std
        _, mask = cv.threshold(satim, new_thresh, 0xff, cv.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8,
                                                                               ltype=cv.CV_32S)
        label_areas = [(i, stats[i, cv.CC_STAT_AREA]) for i in range(num_labels)]
        label_areas.sort(key=lambda p: -p[1])
        mask[(labels != label_areas[0][0]) & (labels != label_areas[1][0])] = 0
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
        # mask = cv.medianBlur(mask, 3) # Might've been more like 11, 13, 15
        return mask

    @cached_property
    def centroid(self) -> Tuple[int, int]:
        moments = cv.moments(self.mask)
        centroid = np.round([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        return int(centroid[0]), int(centroid[1])

    @cached_property
    def primary_axis(self) -> np.array:
        points = np.argwhere(self.mask == 0xff)
        pca = PCA(n_components=2)
        pca.fit(points)
        ax = pca.components_[0]
        ax = ax / np.linalg.norm(ax)
        return np.flip(ax)

    @cached_property
    def normalized_mask(self) -> np.array:
        height, width = self.mask.shape
        pad = max(height, width)
        adj_dim = (height + 2 * pad, width + 2 * pad)
        result = np.zeros(adj_dim, np.uint8)
        result[pad: pad + height, pad:pad + width] = self.mask[:, :]
        ang = min(
            angle_between(self.primary_axis, np.array([1, 0])),
            angle_between(self.primary_axis, np.array([-1, 0])),
            key=lambda a: abs(a))
        adj_centroid = (self.centroid[0] + pad, self.centroid[1] + pad)
        rot = cv.getRotationMatrix2D(adj_centroid, -ang, 1)
        result = cv.warpAffine(result, rot, np.flip(adj_dim))
        if self.side == "right":
            result = cv.flip(result, 1)
        scale_factor = self.calc_avg_scale() / self.scale
        result = cv.resize(result, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)
        _, result = cv.threshold(result, 127, 255, cv.THRESH_BINARY)
        return result

    @cached_property
    def normalized_outline(self) -> np.array:
        contours, _ = cv.findContours(self.normalized_mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        outline = max(contours, key=cv.contourArea)
        outline = [(pt[0][0], pt[0][1]) for pt in outline]
        height, width = self.normalized_mask.shape
        assert all((0 not in pt) and (pt[1] != height - 1) and (pt[0] != width - 1) for pt in outline), \
            "Inadmissible fish; expand ROI"
        minx = min([p[0] for p in outline])
        target_origin = min([p for p in outline if p[0] == minx], key=lambda p: p[1])
        return np.roll(outline, -outline.index(target_origin), axis=0)

    @cached_property
    def signal(self) -> List[float]:
        result = []
        prev_delta = self.normalized_outline[1] - self.normalized_outline[0]
        for i, pt in enumerate(self.normalized_outline[:-1]):
            nextpt = self.normalized_outline[i + 1]
            delta = nextpt - pt
            result.append(angle_between(delta, prev_delta))
            prev_delta = delta
        return result

    @cached_property
    def features(self):
        return extract_k_most_significant_frequencies(self.signal, self.feature_count)

    @cached_property
    def complex_features(self):
        complex_outline = np.empty(self.normalized_outline.shape[:-1], dtype=complex)
        complex_outline.real = self.normalized_outline[:, 0]
        complex_outline.imag = self.normalized_outline[:, 1]
        return extract_k_most_significant_frequencies(complex_outline, self.feature_count)

    def show(self):
        plt.imshow(self.cropped_im)
        plt.show()

    def show_ax(self):
        im = self.cropped_im.copy()
        cv.line(im, self.centroid,
                (self.centroid + np.round(self.primary_axis * self.cropped_im.shape[0])).astype(int), (0, 0xff, 0),
                thickness=2)
        cv.circle(im, self.centroid, 5, (0, 0, 0xff), thickness=-1)
        plt.imshow(im)
        plt.show()

    def show_outline(self):
        mask = cv.cvtColor(self.normalized_mask.copy(), cv.COLOR_GRAY2RGB)
        cv.drawContours(mask, [self.normalized_outline], -1, (0, 0xff, 0), thickness=2)
        plt.imshow(mask)
        plt.show()


if __name__ == "__main__":
    pass
