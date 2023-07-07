from inhs_outlining import *

import multiprocessing as mp
import pathlib
import random


def get_fifteen_species():
    target_species = [species
                      for species, count in Fish.count_fish_per_species().items()
                      if count >= 100]
    target_fish = []
    for species in target_species:
        all_of_species = Fish.all_of_species(*species.split(' '))
        random.shuffle(all_of_species)
        target_fish += all_of_species[:100]
    return target_fish


def get_seven_genera(max_fish_per_genus):
    target_genera = [genus
                     for genus, count in Fish.count_fish_per_genus().items()
                     if count >= 153]
    target_fish = []
    for genus in target_genera:
        all_of_genus = Fish.all_of_genus(genus)
        random.shuffle(all_of_genus)
        target_fish += all_of_genus[:max_fish_per_genus]
    return target_fish


def generate_frag(frags_dir, fishes, procindex):
    pathlib.Path(frags_dir).mkdir(exist_ok=True)
    with open(f"{frags_dir}/{procindex}", 'w') as frag:
        for fish in fishes:
            efds, locus = fish.encoding
            row = np.append(locus, efds.ravel()).astype(str)
            frag.write(','.join(row) + f",{fish.genus} {fish.species}\n")
            del fish


def generate_frags(frags_dir, fishes):
    nproc = mp.cpu_count()
    groups = [list(fish_group) for fish_group in np.array_split(fishes, nproc)]
    processes = []
    for n in range(nproc):
        p = mp.Process(target=generate_frag, args=(frags_dir, groups[n], n))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def build_mat_from_frags(name, frags_dir):
    mat = []
    labels = []
    for fragpath in pathlib.Path(frags_dir).iterdir():
        with open(fragpath, 'r') as frag:
            for row in frag.readlines():
                cols = row.split(',')
                mat.append(cols[:-1])
                labels.append(cols[-1].strip())
    mat = pad_ragged(mat)
    mat = np.concatenate((mat, np.array(labels).reshape(-1, 1)), axis=1)
    np.savetxt(f"{name}.csv", mat, fmt='%s', delimiter=',')
    shutil.rmtree(frags_dir)


def generate_dataset(name, fishes):
    frags_dir = f"{name}_frags/"
    generate_frags(frags_dir, fishes)
    build_mat_from_frags(name, frags_dir)


if __name__ == "__main__":
    generate_dataset("1mm_fifteen_species", get_fifteen_species())
    generate_dataset("1mm_seven_genera", get_seven_genera(153))
    generate_dataset("1mm_aug_seven_genera", get_seven_genera(250))
