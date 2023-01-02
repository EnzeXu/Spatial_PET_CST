import numpy as np
import itertools
from tqdm import tqdm
from ode_truth import ConstTruthSpatial, loss_func_spatial

unit = 86400 * 365 * 1e-12
SPATIAL_DIFFUSION_CONST = [
    {
        "id": 0,
        "name": "d_Am",
        "init": 3.0 * 10 * unit,
        "lb": 1.0 * 10 * unit,
        "ub": 8.0 * 10 * unit,
    },
    {
        "id": 1,
        "name": "d_Ao",
        "init": 0.4 * unit,
        "lb": 0.2 * unit,
        "ub": 0.8 * unit,
    },
    {
        "id": 2,
        "name": "d_Tm",
        "init": 3.0 * unit,
        "lb": 1.5 * unit,
        "ub": 6.0 * unit,
    },
    {
        "id": 3,
        "name": "d_Tp",
        "init": 11.0 * unit,
        "lb": 5.5 * unit,
        "ub": 22.0 * unit,
    },
    {
        "id": 4,
        "name": "d_To",
        "init": 0.075 * unit,
        "lb": 0.05 * unit,
        "ub": 0.1 * unit,
    }
]

def spatial_simulation(split_n=10):
    ct = ConstTruthSpatial(
        csf_folder_path="data/CSF/",
        pet_folder_path="data/PET/",
        # dataset="all"
    )
    save = np.load("saves/params_A2_2250.npy")
    params = save[:46]
    starts = save[-11:]
    diffusion_cuts = [np.linspace(SPATIAL_DIFFUSION_CONST[i]["lb"], SPATIAL_DIFFUSION_CONST[i]["ub"], split_n) for i in range(5)]
    diffusion_cuts = itertools.product(*diffusion_cuts)
    element_id = 0
    best_loss = 999999.0
    best_time_string = None
    for element in tqdm(diffusion_cuts, total=split_n**5):
        # print(element)
        element_id += 1
        diffusion_list = np.asarray(element)
        time_string, loss = loss_func_spatial(params, starts, diffusion_list, ct, True, False, element_id)
        if loss < best_loss:
            best_time_string = time_string
            best_loss = loss

if __name__ == "__main__":
    spatial_simulation(10)
