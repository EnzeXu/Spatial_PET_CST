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
    from const import PARAM_NUM, STARTS_NUM
    ct = ConstTruthSpatial(
        csf_folder_path="data/CSF/",
        pet_folder_path="data/PET/",
        # dataset="all"
    )
    save = np.load("saves/params_20230113_102137_344977.npy")
    print("overall input parameter length: {}".format(len(save)))
    params = save[:PARAM_NUM]
    starts = save[-STARTS_NUM:]
    diffusion_cuts = [np.linspace(SPATIAL_DIFFUSION_CONST[i]["lb"], SPATIAL_DIFFUSION_CONST[i]["ub"], split_n) for i in range(5)]
    diffusion_cuts = itertools.product(*diffusion_cuts)
    element_id = 0
    best_loss = 999999.0
    best_element_id = -1
    best_time_string = None
    save_file_path = "figure_spatial/record_20230130.csv"
    for element in tqdm(diffusion_cuts, total=split_n**5):
        # print(element)
        element_id += 1
        diffusion_list = np.asarray(element)
        time_string, loss = loss_func_spatial(params, starts, diffusion_list, ct, save_file_path, silent=True, save_flag=True, element_id=element_id)
        if loss < best_loss:
            best_time_string = time_string
            best_loss = loss
            best_element_id = element_id
    print("best_element_id: {} best_time_string: {} best_loss: {}".format(best_element_id, best_time_string, best_loss))
    with open(save_file_path, "a") as f:
        f.write("{0},{1},{2}\n".format(best_element_id, best_time_string, best_loss))

if __name__ == "__main__":
    spatial_simulation(5)
