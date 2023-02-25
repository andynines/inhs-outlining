from pathlib import Path
import socket
from sqlalchemy import LargeBinary, String, Float, create_engine, select, distinct
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from werkzeug.utils import cached_property
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from typing import List, Tuple


def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.rad2deg(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))


def assert_is_lab_server():
    assert socket.gethostname() == "CCI-DX4M513", "Not on lab server"


class Base(DeclarativeBase):
    pass


class Fish(Base):
    __tablename__ = "fish"

    engine = create_engine("sqlite:///fish.db")

    @classmethod
    def query(cls, stmt):
        with Session(cls.engine) as session:
            return session.scalars(stmt).all()

    @classmethod
    def with_id(cls, fid: str):
        return cls.query(select(cls).where(cls.id == fid))[0]

    @classmethod
    def all(cls):
        return cls.query(select(cls))

    # IDs aren't purely numeric! Some have underscores in them.
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    genus: Mapped[str] = mapped_column(String(50))
    species: Mapped[str] = mapped_column(String(50))
    image: Mapped[bytes] = mapped_column(LargeBinary)
    side: Mapped[str] = mapped_column(String(10))
    scale: Mapped[float] = mapped_column(Float)

    def __repr__(self) -> str:
        return f"<INHS_FISH_{self.id}>"

    @cached_property
    def cropped_im(self) -> np.array:
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
    def primary_axis(self) -> np.array:
        points = np.argwhere(self.mask == 0xff)
        pca = PCA(n_components=2)
        pca.fit(points)
        ax = pca.components_[0]
        ax = ax / np.linalg.norm(ax)
        return np.flip(ax)

    @cached_property
    def normalized_mask(self) -> np.array:
        pad = 50  # TODO: calculate exactly what this should be to prevent any fish pixel from rotating out of bounds. It's a function of the image dimensions.
        height, width = self.mask.shape
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
    def centroid(self) -> Tuple[int, int]:
        moments = cv.moments(self.mask)
        centroid = np.round([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        return int(centroid[0]), int(centroid[1])

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

    def show_ax(self) -> None:
        im = self.cropped_im.copy()
        cv.line(im, self.centroid,
                (self.centroid + np.round(self.primary_axis * self.cropped_im.shape[0])).astype(int), (0, 0xff, 0),
                thickness=2)
        cv.circle(im, self.centroid, 5, (0, 0, 0xff), thickness=-1)
        plt.imshow(im)
        plt.show()


if __name__ == "__main__":
    fish = Fish.with_id("45942")
    plt.imshow(fish.normalized_mask)
    plt.show()


"""
def classify(data):
    np.random.shuffle(data)
    test = data[:len(data)//3]
    train = data[len(data)//3:]
    clf = svm.SVC(kernel="linear")
    clf.fit(train[:, :3], train[:, 3])
    predictions = clf.predict(test[:, :3])
    expecteds = test[:, 3]
    print("Accuracy:", np.sum(predictions == expecteds) / len(predictions))
"""

"""
def create_db() -> None:
    NAME_CSV = Path("./ml-ready.csv")
    METADATA_JSON = Path("./ml-ready-metadata.json")
    BBOX_PAD_PX = 10

    engine = create_engine(DB, echo=True)
    Base.metadata.create_all(engine)

    with open(NAME_CSV, 'r') as name_csv_file:
        name_csv = [row for row in csv.reader(name_csv_file)]

    with open(METADATA_JSON, 'r') as metadata_json_file:
        metadata_json = json.load(metadata_json_file)

    with Session(engine) as session:
        for row in name_csv[1:]:
            inhs_id = row[1].replace(".jpg", "")
            print(inhs_id, end=' ')
            if not metadata_json[inhs_id]["has_fish"]:
                continue
            fid = inhs_id.replace("INHS_FISH_", "")
            genus = row[2]
            species = row[3]
            bbox = metadata_json[inhs_id]["fish"][0]["bbox"]
            img = cv.imread(str(INHSFish.find_image(fid)))
            cropped_img = img[
                          max(0, bbox[1] - BBOX_PAD_PX): bbox[3] + BBOX_PAD_PX,
                          max(0, bbox[0] - BBOX_PAD_PX): bbox[2] + BBOX_PAD_PX,
                          ]
            new_fish_record = INHSFish(
                id=fid,
                genus=genus,
                species=species,
                image=cv.imencode('.jpg', cropped_img)[1].tobytes(),
            )
            session.add(new_fish_record)
        session.commit()
"""
