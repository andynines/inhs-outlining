# INHS Fish Outlining

![Example fish outline](example.png)

## Setup
###1.
Ensure you have the latest version of `fish.db`. I've reached my free Git LFS quota, so the copy included here may be old. The latest version has this SHA1 hash:
```
$ sha1sum fish.db
c541f5aeb8f5d5ad3cf9b01e63c18cb35768fbb5  fish.db
```
If your version differs, please download the latest:
```
wget http://andrewsenin.com/fish.db
```
###2.
Create a __Python 3__ virtual environment in the top level of the repository:
```
$ python -m venv venv/
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

## Reproducing presentation results
###1.
Generate the dataset. This takes around ten minutes on eight, Intel i5 cores at 3.4 GHz:
```
python datagen.py
```
This produces the file `1mm_fifteen_species.csv`.
###2.
Run the classification experiment:
```
python classif.py
```
You may observe higher accuracy than I reported during the presentation. I've since begun normalizing the elliptic Fourier descriptors' scale, which gives a slight accuracy boost.

## Outstanding issues
* Morphing animations between fish (not shown in the presentation) are a work in progress. The function `animate_morph_between()` in `inhs_outlining.py` produces a GIF, but it's choppy.
* There are several fish that are close enough to other objects in the image for their outlines to connect. I have a fix but haven't implemented it yet.
