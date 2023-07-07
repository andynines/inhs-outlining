
from werkzeug.utils import cached_property
from PIL import Image as PILImage
import pathlib
import shutil

from sqlalchemy import LargeBinary, String, Float, Integer, create_engine, select
from sqlalchemy.sql import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
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


def make_contour_im(contour, *additional_contours, colors=None):
    if colors is None:
        colors = [(0x7f, 0x7f, 0x7f) for _ in range(len(additional_contours))] + [(0xff, 0xff, 0xff)]
    mins = abs(np.min(contour, axis=0))
    maxes = np.max(contour, axis=0)
    pad = 10
    dim = np.flip(mins + maxes) + 2 * pad
    dim = np.concatenate((dim, (3,)))
    im = np.zeros(dim)
    citer = iter(colors)
    for addl in additional_contours:
        im = cv.drawContours(im, [addl + mins + (pad, pad)], -1, next(citer), thickness=1)
    im = cv.drawContours(im, [contour + mins + (pad, pad)], -1, next(citer), thickness=1)
    return im.astype(int)


def show_contour(contour, *additional_contours, colors=None):
    im = make_contour_im(contour, *additional_contours, colors=colors)
    showim(im)


def contour_error(contour1, contour2):
    return max(
        directed_hausdorff(contour1, contour2)[0],
        directed_hausdorff(contour2, contour1)[0]
    )


def encode(contour, num_harmonics):
    return pyefd.elliptic_fourier_descriptors(contour, order=num_harmonics, normalize=False), \
           pyefd.calculate_dc_coefficients(contour)


def reconstruct(efds, num_points, locus):
    reconstruction = pyefd.reconstruct_contour(efds, num_points=num_points, locus=locus)
    reconstruction = np.round(reconstruction).astype(int)
    return reconstruction


def pad_ragged(mat):
    max_row_len = max(len(row) for row in mat)
    for row in mat:
        row += [0] * (max_row_len - len(row))
    return np.array(mat)


def make_encoding_mat(fishes):
    encodings = [fish.encoding[0] for fish in fishes]
    return pad_ragged([list(efds.ravel()) for efds in encodings])


def cross(fishes, weights=None):
    if weights is None:
        n = len(fishes)
        weights = [1 / n for _ in range(n)]
    weights = np.array(weights).reshape(-1, 1)
    encoding_mat = make_encoding_mat(fishes)
    result = np.sum(encoding_mat * weights, axis=0)
    return result.reshape(-1, 4)


def synthesize_fish_from(fishes):
    n = np.random.randint(1, len(fishes))
    np.random.shuffle(fishes)
    choices = fishes[:n]
    weights = np.random.dirichlet(np.ones(n), size=1)
    return cross(choices, weights=weights)


def animate_morph_between(fish1, fish2, n_frames=50, speed=0.3, num_points=300):
    # Broken!
    frames = []
    for i, w in enumerate(np.linspace(0, 1, n_frames)):
        efds = cross((fish1, fish2), weights=(1 - w, w))
        contour = reconstruct(efds, num_points, (0, 0))
        frame = make_contour_im(contour)
        frames.append(frame)
    frame_dims = np.array([frame.shape for frame in frames])
    frame_dim_max = np.array([np.max(frame_dims[:, 0]), np.max(frame_dims[:, 1])])
    frame_dir = pathlib.Path(f"./giftemp{fish1.id}to{fish2.id}/")
    frame_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        dy, dx = frame_dim_max - frame.shape
        padded = cv.copyMakeBorder(frame, top=dy // 2, bottom=dy // 2 + (dy % 2), left=dx // 2,
                                   right=dx // 2 + (dx % 2),
                                   borderType=cv.BORDER_CONSTANT, value=0)
        cv.imwrite(str(frame_dir / f"frame{i}.png"), padded)
    del frames
    frames = [PILImage.open(str(imf)).convert('P') for imf in frame_dir.iterdir()]
    gif_name = f"{fish1.id}-to-{fish2.id}-{n_frames}f-{speed}spd.gif"
    frames[0].save(gif_name, format="GIF", append_images=frames, save_all=True, duration=500, loop=0, optimize=False)
    shutil.rmtree(frame_dir)


class Base(DeclarativeBase):
    pass


class Fish(Base):
    __tablename__ = "fish"

    engine = create_engine("sqlite:///fish.db")

    bbox_pad_mult = 0.05
    spatial_resolution = 40  # The average of all records in fish.db is just under 76 px/cm.
    dark_thresh_mult = 0.5
    close_kern_size = 5
    close_iters = 2
    scl_interp_method = cv.INTER_CUBIC
    reconstruction_tol = 0.1 * spatial_resolution  # px
    harmonics_limit = 100

    @classmethod
    def show_params(cls):
        just = 40
        print(
            "Bounding box padding percentage:".ljust(just) + f"{cls.bbox_pad_mult * 100}%",
            "Spatial resolution:".ljust(just) + f"{cls.spatial_resolution} px/cm",
            "Dark range std multiplier:".ljust(just) + str(cls.dark_thresh_mult),
            "Closing kernel size:".ljust(just) + f"{cls.close_kern_size}x{cls.close_kern_size} px",
            "Closing iterations:".ljust(just) + str(cls.close_iters),
            "Scaling interpolation method (CV enum):".ljust(just) + str(cls.scl_interp_method),
            "Reconstruction tolerance:".ljust(just) + f"{cls.reconstruction_tol} px",
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
    def show_fish_per_genus(cls):
        genus_counts = list(cls.count_fish_per_genus().items())
        genus_counts.sort(key=lambda p: -p[1])
        genera, counts = zip(*genus_counts)
        plt.bar(range(len(genera)), counts, align="center")
        plt.xticks(range(len(genera)), genera, rotation=45)
        showplt()

    @classmethod
    def count_fish_per_species(cls):
        counts = cls.sesh(
            lambda s: s.query(cls.genus, cls.species, func.count(cls.id)).group_by(cls.genus, cls.species).all())
        counts.sort(key=lambda count: -count[2])
        return {f"{count[0]} {count[1]}": count[2] for count in counts}

    @classmethod
    def example_of(cls, genus, species):
        return cls.all_of_species(genus, species)[0]

    @classmethod
    def all_of_genus(cls, genus):
        return cls.query(select(cls).where(cls.genus == genus))

    @classmethod
    def all_of_species(cls, genus, species):
        return cls.query(select(cls).where((cls.genus == genus) & (cls.species == species)))

    @classmethod
    def contour_area(cls, contour):  # cm^2
        return cv.contourArea(contour) / cls.spatial_resolution ** 2

    @classmethod
    def contour_perimeter(cls, contour):  # cm
        return cv.arcLength(contour, closed=True) / cls.spatial_resolution

    # IDs aren't purely numeric! Some have underscores in them, so the column type must be str
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    genus: Mapped[str] = mapped_column(String(50))
    species: Mapped[str] = mapped_column(String(50))
    image: Mapped[bytes] = mapped_column(LargeBinary)  # = cv.imencode('.jpg', img)[1].tobytes()
    side: Mapped[str] = mapped_column(String(10))
    scale: Mapped[float] = mapped_column(Float)

    fish_bbox_ul_x: Mapped[int] = mapped_column(Integer)
    fish_bbox_ul_y: Mapped[int] = mapped_column(Integer)
    fish_bbox_lr_x: Mapped[int] = mapped_column(Integer)
    fish_bbox_lr_y: Mapped[int] = mapped_column(Integer)

    label_bbox_ul_x: Mapped[int] = mapped_column(Integer)
    label_bbox_ul_y: Mapped[int] = mapped_column(Integer)
    label_bbox_lr_x: Mapped[int] = mapped_column(Integer)
    label_bbox_lr_y: Mapped[int] = mapped_column(Integer)

    def __repr__(self):
        return f"INHS_FISH_{self.id}"

    def __str__(self):
        return repr(self)

    @property
    def cropped_im(self):
        crop = self.original_im.copy()
        pad_x = round((self.fish_bbox_lr_x - self.fish_bbox_ul_x) * self.bbox_pad_mult)
        pad_y = round((self.fish_bbox_lr_y - self.fish_bbox_ul_y) * self.bbox_pad_mult)
        return crop[
               max(0, self.fish_bbox_ul_y - pad_y): self.fish_bbox_lr_y + pad_y,
               max(0, self.fish_bbox_ul_x - pad_x): self.fish_bbox_lr_x + pad_x,
               ]

    @property
    def saturation_im(self):
        hsv = cv.cvtColor(self.cropped_im, cv.COLOR_RGB2HSV)
        return hsv[:, :, 1]

    @property
    def original_im(self):
        nparr = np.frombuffer(self.image, np.uint8)
        im = cv.imdecode(nparr, cv.IMREAD_COLOR)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)

    @cached_property
    def mask(self):
        otsu_thresh, _ = cv.threshold(self.saturation_im, 0, 0xff, cv.THRESH_BINARY | cv.THRESH_OTSU)
        dark_px = self.saturation_im[self.saturation_im < otsu_thresh].ravel()
        dark_mean = np.nanmean(dark_px)
        dark_std = np.nanstd(dark_px)
        new_thresh = dark_mean + self.dark_thresh_mult * dark_std
        _, mask = cv.threshold(self.saturation_im, new_thresh, 0xff, cv.THRESH_BINARY)
        # Black out the info card so if the fish overlaps it, it doesn't become part of the fish's outline
        # This cuts a chunk out of the fish, but the morphological closing below will make it look slightly better
        # We haven't seen any reason to black out rulers
        mask[self.label_bbox_ul_y-self.fish_bbox_ul_y:self.label_bbox_lr_y-self.fish_bbox_ul_y,
             self.label_bbox_ul_x-self.fish_bbox_ul_x:self.label_bbox_lr_x-self.fish_bbox_ul_x] = 0
        # Remove all connected components but the largest two, which will be the fish and the background
        num_labels, labels, stats, _ = \
            cv.connectedComponentsWithStats(mask, connectivity=8, ltype=cv.CV_32S)
        label_areas = [(i, stats[i, cv.CC_STAT_AREA]) for i in range(num_labels)]
        label_areas.sort(key=lambda p: -p[1])
        mask[(labels != label_areas[0][0]) & (labels != label_areas[1][0])] = 0
        # Perform morphological closing to close holes and smooth edges
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
        """
        We've already computed this in Pepper, Karnani et. al. But I have my own version using the same method as before
        because it uses the fish mask, and my masks are more accurate. This is also why I have my own functions for
        area, perimeter, etc.
        """
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
        # Pad image generously so that no parts of the fish get clipped off during rotation
        result[pad: pad + height, pad:pad + width] = self.mask[:, :]
        # Rotate the fish so it faces straight to the side
        ang = min(
            angle_between(self.primary_axis, np.array([1, 0])),
            angle_between(self.primary_axis, np.array([-1, 0])),
            key=lambda a: abs(a))
        adj_centroid = (self.centroid[0] + pad, self.centroid[1] + pad)
        rot = cv.getRotationMatrix2D(adj_centroid, -ang, 1)
        result = cv.warpAffine(result, rot, np.flip(adj_dim))
        if self.side == "right":
            result = cv.flip(result, 1)
        # Normalize image scale to Fish.spatial_resolution
        scale_factor = self.spatial_resolution / self.scale
        result = cv.resize(result, None, fx=scale_factor, fy=scale_factor, interpolation=self.scl_interp_method)
        _, result = cv.threshold(result, 127, 255, cv.THRESH_BINARY)
        return result

    @property
    def area(self):  # cm^2
        return self.contour_area(self.normalized_outline)

    @property
    def perimeter(self):  # cm
        return self.contour_perimeter(self.normalized_outline)

    @cached_property
    def normalized_outline(self):
        contours, _ = cv.findContours(self.normalized_mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        outline = max(contours, key=cv.contourArea)
        outline = [(pt[0][0], pt[0][1]) for pt in outline]
        # Shift the sequence of coordinates until it begins with the highest leftmost point
        minx = min([p[0] for p in outline])
        target_origin = min([p for p in outline if p[0] == minx], key=lambda p: p[1])
        outline = np.roll(outline, -outline.index(target_origin), axis=0)
        # Center outline around (0, 0) and round its coordinates
        outline = np.array(outline) - np.mean(outline, axis=0)
        outline = np.round(outline).astype(int)
        # Remove duplicate successive points
        # We wait until now to remove them because rounding can create more duplicates
        return np.array([outline[i] for i in range(len(outline)) if i == 0 or (outline[i] != outline[i - 1]).any()])

    @cached_property
    def encoding(self):
        num_points = self.normalized_outline.shape[0]
        num_harmonics = 1
        while num_harmonics <= self.harmonics_limit:
            efds, locus = encode(self.normalized_outline, num_harmonics)
            reconstruction = reconstruct(efds, num_points, locus)
            if contour_error(self.normalized_outline, reconstruction) <= self.reconstruction_tol:
                return efds, locus
            num_harmonics += 1
        raise AssertionError(f"Failed to fit within tolerance with {self.harmonics_limit} harmonics")

    @cached_property
    def reconstruction(self):
        efds, locus = self.encoding
        return reconstruct(efds, self.normalized_outline.shape[0], locus)

    def show(self):
        showim(self.cropped_im)

    def show_saturation_hist(self):
        plt.figure()
        hist = plt.hist(self.saturation_im.ravel(), 256, [0, 256])
        plt.xlabel("Intensity")
        plt.ylabel("Pixels")
        plt.margins(x=0)
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

    def show_reconstruction(self):
        show_contour(self.reconstruction)

    def save(self):
        cv.imwrite(repr(self) + ".png", cv.cvtColor(self.cropped_im, cv.COLOR_RGB2BGR))


if __name__ == "__main__":
    pass
