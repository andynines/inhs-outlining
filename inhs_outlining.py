from pathlib import Path
import socket
from werkzeug.utils import cached_property

from sqlalchemy import LargeBinary, String, Float, create_engine, select
from sqlalchemy.sql import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import directed_hausdorff

import pyefd


def showplt():
    plt.show(block=False)


def showim(im, gray=False):
    plt.figure()
    plt.imshow(im, **({"cmap": "gray"} if gray else {}))
    showplt()


def closeplt():
    plt.close("all")


def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.rad2deg(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))


def show_contour(contour, *additional_contours):
    mins = abs(np.min(contour, axis=0))
    maxes = np.max(contour, axis=0)
    pad = 2
    im = np.zeros(np.flip(mins + maxes) + 2 * pad)
    for i, addl in enumerate(additional_contours):
        c = 0xff / (i + 2)
        im = cv.drawContours(im, [addl + mins + (pad, pad)], -1, (c, c, c), thickness=1)
    im = cv.drawContours(im, [contour + mins + (pad, pad)], -1, (0xff, 0xff, 0xff), thickness=1)
    showim(im, gray=True)


def contour_error(contour1, contour2):
    return max(
        directed_hausdorff(contour1, contour2)[0],
        directed_hausdorff(contour2, contour1)[0]
    )


def encode(contour, num_harmonics):
    return pyefd.elliptic_fourier_descriptors(contour, order=num_harmonics, normalize=False)


def reconstruct(efds, num_points):
    reconstruction = pyefd.reconstruct_contour(efds, num_points=num_points)
    reconstruction = np.round(reconstruction).astype(int)
    return reconstruction


def assert_is_lab_server():
    assert socket.gethostname() == "CCI-DX4M513", "Not on lab server"


class Base(DeclarativeBase):
    pass


class Fish(Base):
    __tablename__ = "fish"

    engine = create_engine("sqlite:///fish.db")

    breen_feature_count = 40
    target_scale = 40  # The average of all records in fish.db is just under 76 px/cm.
    outline_connectivity = 4
    dark_thresh_mult = 0.5
    close_kern_size = 5
    close_iters = 2
    scl_interp_method = cv.INTER_CUBIC
    reconstruction_tol = 0.1 * target_scale
    harmonics_limit = 100

    @classmethod
    def show_params(cls):
        just = 40
        print(
            "Target scale:".ljust(just) + f"{cls.target_scale} px/cm",
            "Outline connectivity:".ljust(just) + f"{cls.outline_connectivity}-way",
            "Dark range std multiplier:".ljust(just) + str(cls.dark_thresh_mult),
            "Closing kernel size:".ljust(just) + f"{cls.close_kern_size}x{cls.close_kern_size}",
            "Closing iterations:".ljust(just) + str(cls.close_iters),
            "Scaling interpolation method (CV enum):".ljust(just) + str(cls.scl_interp_method),
            "Reconstruction tolerance:".ljust(just) + f"{cls.reconstruction_tol} px",
            "Harmonics limit:".ljust(just) + str(cls.harmonics_limit),
            sep='\n'
        )

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
        return cls.all_of_species(genus, species)[0]

    @classmethod
    def all_of_species(cls, genus, species):
        return cls.query(select(cls).where((cls.genus == genus) & (cls.species == species)))

    # IDs aren't purely numeric! Some have underscores in them.
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    genus: Mapped[str] = mapped_column(String(50))
    species: Mapped[str] = mapped_column(String(50))
    image: Mapped[bytes] = mapped_column(LargeBinary)  # = cv.imencode('.jpg', img)[1].tobytes(),
    side: Mapped[str] = mapped_column(String(10))
    scale: Mapped[float] = mapped_column(Float)

    def __repr__(self):
        return f"INHS_FISH_{self.id}"

    def __str__(self):
        return repr(self)

    @cached_property
    def cropped_im(self):
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
    def saturation_im(self):
        hsv = cv.cvtColor(self.cropped_im, cv.COLOR_RGB2HSV)
        return hsv[:, :, 1]

    @cached_property
    def original_im(self):
        assert_is_lab_server()
        path_template = f"/usr/local/bgnn/inhs_{{group}}/INHS_FISH_{self.id}.jpg"
        validation_path = Path(path_template.format(group="validation"))
        if validation_path.exists():
            im = cv.imread(validation_path)
        else:
            im = cv.imread(Path(path_template.format(group="test")))
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)

    @cached_property
    def mask(self):
        otsu_thresh, _ = cv.threshold(self.saturation_im, 0, 0xff, cv.THRESH_BINARY | cv.THRESH_OTSU)
        dark_px = self.saturation_im[self.saturation_im < otsu_thresh].ravel()
        dark_mean = np.mean(dark_px)
        dark_std = np.std(dark_px)
        new_thresh = dark_mean + self.dark_thresh_mult * dark_std
        _, mask = cv.threshold(self.saturation_im, new_thresh, 0xff, cv.THRESH_BINARY)
        num_labels, labels, stats, _ = \
            cv.connectedComponentsWithStats(mask, connectivity=self.outline_connectivity, ltype=cv.CV_32S)
        label_areas = [(i, stats[i, cv.CC_STAT_AREA]) for i in range(num_labels)]
        label_areas.sort(key=lambda p: -p[1])
        mask[(labels != label_areas[0][0]) & (labels != label_areas[1][0])] = 0
        kernel = np.ones((self.close_kern_size, self.close_kern_size), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=self.close_iters)
        return mask

    @cached_property
    def centroid(self):
        moments = cv.moments(self.mask)
        result = np.round([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        return int(result[0]), int(result[1])

    @cached_property
    def primary_axis(self):
        points = np.argwhere(self.mask == 0xff)
        pca = PCA(n_components=2)
        pca.fit(points)
        ax = pca.components_[0]
        ax = ax / np.linalg.norm(ax)
        return np.flip(ax)

    @cached_property
    def normalized_mask(self):
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
        scale_factor = self.target_scale / self.scale
        result = cv.resize(result, None, fx=scale_factor, fy=scale_factor, interpolation=self.scl_interp_method)
        _, result = cv.threshold(result, 127, 255, cv.THRESH_BINARY)
        return result

    @property
    def area(self):  # cm^2
        return np.sum(self.normalized_mask) / 0xff / self.target_scale ** 2

    @cached_property
    def normalized_outline(self):
        contours, _ = cv.findContours(self.normalized_mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        outline = max(contours, key=cv.contourArea)
        outline = np.array([(pt[0][0], pt[0][1]) for pt in outline])
        height, width = self.normalized_mask.shape
        assert all((0 not in pt) and (pt[1] != height - 1) and (pt[0] != width - 1) for pt in outline), \
            "Inadmissible fish touches image border; expand ROI"
        """
        # Shift the sequence of coordinates until it begins with the highest leftmost point
        minx = min([p[0] for p in outline])
        target_origin = min([p for p in outline if p[0] == minx], key=lambda p: p[1])
        result = np.roll(outline, -outline.index(target_origin), axis=0)
        return result - result[0]
        """
        outline_centroid = np.mean(outline, axis=0)
        return np.round(outline - outline_centroid).astype(int)

    @cached_property
    def breen_features(self):
        angle_signal = []
        prev_delta = self.normalized_outline[1] - self.normalized_outline[0]
        for i, pt in enumerate(self.normalized_outline[:-1]):
            nextpt = self.normalized_outline[i + 1]
            delta = nextpt - pt
            angle_signal.append(angle_between(delta, prev_delta))
            prev_delta = delta
        n = len(angle_signal)
        dft = list(zip(np.fft.fft(angle_signal), n * np.fft.fftfreq(n)))
        highest_term_freqs = sorted(dft[1:n // 2], key=lambda p: -p[0])[:self.breen_feature_count]
        return np.round(np.array(highest_term_freqs)[:, 1])

    @cached_property
    def efds(self):
        num_points = self.normalized_outline.shape[0]
        num_harmonics = 1
        while num_harmonics <= self.harmonics_limit:
            efds = encode(self.normalized_outline, num_harmonics)
            reconstruction = reconstruct(efds, num_points)
            if contour_error(self.normalized_outline, reconstruction) <= self.reconstruction_tol:
                return efds
            num_harmonics += 1
        raise AssertionError(f"Failed to fit within tolerance with {self.harmonics_limit} harmonics")

    @cached_property
    def reconstruction(self):
        return reconstruct(self.efds, self.normalized_outline.shape[0])

    def show(self):
        showim(self.cropped_im)

    def show_saturation_hist(self):
        hist = plt.hist(self.saturation_im.ravel(), 256, [0, 256])
        showplt()
        return hist

    def show_ax(self):
        im = self.cropped_im.copy()
        cv.line(im, self.centroid,
                (self.centroid + np.round(self.primary_axis * self.cropped_im.shape[0])).astype(int), (0, 0xff, 0),
                thickness=2)
        cv.circle(im, self.centroid, 5, (0, 0, 0xff), thickness=-1)
        showim(im)

    def show_outline(self):
        show_contour(self.normalized_outline)

    def save(self):
        cv.imwrite(repr(self) + ".png", cv.cvtColor(self.cropped_im, cv.COLOR_RGB2BGR))


if __name__ == "__main__":
    pass
