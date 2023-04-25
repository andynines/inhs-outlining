from inhs_outlining import *

import multiprocessing as mp
import pathlib


def get_target_fish():
    # target_species = [species
    #                   for species, count in Fish.count_unique_species().items()
    #                   if count >= 100]
    # target_fish = []
    # for species in target_species:
    #     target_fish += Fish.all_of_species(*species.split(' '))
    return Fish.all()


def list_efds(fish, index):
    failure_ids = []
    pathlib.Path("frags/").mkdir(exist_ok=True)
    pathlib.Path("failures/").mkdir(exist_ok=True)
    with open(f"frags/efdfrag{index}.txt", 'w') as efdfrag:
        for f in fish:
            try:
                efds = f.features.astype(str).ravel()
                efdfrag.write(','.join(efds) + f",{f.genus} {f.species}\n")
            except AssertionError:
                failure_ids.append(f.id)
            del f
    with open(f"failures/failures{index}.txt", 'w') as failures:
        for failure_id in failure_ids:
            failures.write(f"{failure_id}\n")


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
    for fragpath in pathlib.Path("frags/").iterdir():
        with open(fragpath, 'r') as efdfrag:
            mat += [row.split(',') for row in efdfrag.readlines()]
    maxrowlen = max(len(row) for row in mat)
    for row in mat:
        for _ in range(maxrowlen - len(row)):
            row.insert(-1, 0)
    mat = np.array(mat)
    mat[:, -1] = [label.strip() for label in mat[:, -1]]
    np.savetxt("normalized_1mm_feats.csv", mat, fmt='%s', delimiter=',')


if __name__ == "__main__":
    generate_efd_lists(get_target_fish())
    collect_frags()
