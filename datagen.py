from inhs_outlining import *

import multiprocessing as mp
import pathlib
import random


def get_target_fish():
    random.seed(0)
    target_species = [species
                      for species, count in Fish.count_unique_species().items()
                      if count >= 100]
    target_fish = []
    for species in target_species:
        all_of_species = Fish.all_of_species(*species.split(' '))
        random.shuffle(all_of_species)
        target_fish += all_of_species[:100]
    return target_fish


def list_efds(fish, index):
    pathlib.Path("frags/").mkdir(exist_ok=True)
    with open(f"frags/efdfrag{index}.txt", 'w') as efdfrag:
        for f in fish:
            efds, locus = f.encoding
            row = np.append(locus, efds.ravel()).astype(str)
            efdfrag.write(','.join(row) + f",{f.genus} {f.species}\n")
            del f


def generate_efd_lists(fish):
    nproc = mp.cpu_count()
    groups = [list(fish_group) for fish_group in np.array_split(fish, nproc)]
    processes = []
    for n in range(nproc):
        p = mp.Process(target=list_efds, args=(groups[n], n))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def collect_frags():
    mat = []
    labels = []
    for fragpath in pathlib.Path("frags/").iterdir():
        with open(fragpath, 'r') as efdfrag:
            for row in efdfrag.readlines():
                cols = row.split(',')
                mat.append(cols[:-1])
                labels.append(cols[-1].strip())
    mat = pad_ragged(mat)
    mat = np.concatenate((mat, np.array(labels).reshape(-1, 1)), axis=1)
    np.savetxt("1mm_fifteen_species.csv", mat, fmt='%s', delimiter=',')


if __name__ == "__main__":
    generate_efd_lists(get_target_fish())
    collect_frags()
