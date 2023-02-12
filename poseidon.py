import csv
import json
from pathlib import Path
from sqlalchemy import LargeBinary, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
import cv2 as cv
import numpy as np

DB = "sqlite:///fish.db"


class Base(DeclarativeBase):
    pass


class INHSFish(Base):
    __tablename__ = "fish"

    @classmethod
    def with_id(cls, fid: str):
        engine = create_engine(DB)
        with Session(engine) as session:
            stmt = select(INHSFish).where(INHSFish.id == fid)
            return session.scalars(stmt).one()

    # IDs aren't purely numeric! Some have underscores in them.
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    genus: Mapped[str] = mapped_column(String(50))
    species: Mapped[str] = mapped_column(String(50))
    image: Mapped[bytes] = mapped_column(LargeBinary)

    def __repr__(self) -> str:
        return f"<INHS_FISH_{self.id}>"

    def get_cv_img(self) -> np.array:
        nparr = np.fromstring(self.image, np.uint8)
        im = cv.imdecode(nparr, cv.IMREAD_COLOR)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)

    def get_original_img_path(self) -> Path:
        path_template = f"/usr/local/bgnn/inhs_{{group}}/INHS_FISH_{self.id}.jpg"
        validation_path = Path(path_template.format(group="validation"))
        if validation_path.exists():
            return validation_path
        else:
            return Path(path_template.format(group="test"))

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
