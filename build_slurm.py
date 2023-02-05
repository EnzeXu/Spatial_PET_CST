draft = """#!/bin/bash
#SBATCH --job-name="{0}"
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=16GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/{0}-%j.o"
#SBATCH --error="jobs_oe/{0}-%j.e"
echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python {1} {2}
"""



def one_slurm(job_name, python_name, kwargs, draft=draft):
    path = "jobs/{}.slurm".format(job_name)
    print("building {}".format(path))
    with open(path, "w") as f:
        f.write(draft.format(
            job_name,
            python_name,
            " ".join(["--{0} {1}".format(one_key, kwargs[one_key]) for one_key in kwargs]),
        ))



def one_time_build_A():
    plans = [
        # ["A1", 1000, "all", "fixed", 100],
        # ["A1", 2000, "all", "fixed", 100],
        # ["A2", 1000, "all", "ranged", 100],
        # ["A2", 2000, "all", "ranged", 100],
        # ["A3", 1000, "chosen_0", "fixed", 100],
        # ["A3", 2000, "chosen_0", "fixed", 100],
        # ["A4", 1000, "chosen_0", "ranged", 100],
        # ["A4", 2000, "chosen_0", "ranged", 100],
        # ["A1", 1000, "all", "fixed", 100],
        # ["A1", 3000, "all", "fixed", 100],
        # ["A1", 5000, "all", "fixed", 100],
        # ["A1", 7000, "all", "fixed", 100],
        # ["A2", 1000, "all", "ranged", 100],
        # ["A2", 3000, "all", "ranged", 100],
        # ["A2", 5000, "all", "ranged", 100],
        # ["A2", 7000, "all", "ranged", 100],
        # ["A1", 1250, "all", "fixed", 100],
        # ["A1", 1500, "all", "fixed", 100],
        # ["A1", 1750, "all", "fixed", 100],
        # ["A1", 2000, "all", "fixed", 100],
        # ["A2", 1250, "all", "ranged", 100],
        # ["A2", 1500, "all", "ranged", 100],
        # ["A2", 1750, "all", "ranged", 100],
        # ["A2", 2000, "all", "ranged", 100],
        ["A1", 3500, "all", "fixed", 100],
        ["A1", 4000, "all", "fixed", 100],
        ["A1", 4500, "all", "fixed", 100],
        ["A1", 5000, "all", "fixed", 100],
        ["A2", 2250, "all", "ranged", 100],
        ["A2", 2500, "all", "ranged", 100],
        ["A2", 2750, "all", "ranged", 100],
        ["A2", 3000, "all", "ranged", 100],
    ]
    dic = dict()
    for one_plan in plans:
        dic["generation"] = one_plan[1]
        dic["dataset"] = one_plan[2]
        dic["start"] = one_plan[3]
        dic["pop_size"] = one_plan[4]

        one_slurm(
            "GA_{}_{}".format(one_plan[0], one_plan[1]),
            "test_nsga.py",
            dic)

def one_time_build_CD():
    plans = [
        # ["A1", 1000, "all", "fixed", 100],
        # ["A1", 2000, "all", "fixed", 100],
        # ["A2", 1000, "all", "ranged", 100],
        # ["A2", 2000, "all", "ranged", 100],
        # ["A3", 1000, "chosen_0", "fixed", 100],
        # ["A3", 2000, "chosen_0", "fixed", 100],
        # ["A4", 1000, "chosen_0", "ranged", 100],
        # ["A4", 2000, "chosen_0", "ranged", 100],
        # ["A1", 1000, "all", "fixed", 100],
        # ["A1", 3000, "all", "fixed", 100],
        # ["A1", 5000, "all", "fixed", 100],
        # ["A1", 7000, "all", "fixed", 100],
        # ["A2", 1000, "all", "ranged", 100],
        # ["A2", 3000, "all", "ranged", 100],
        # ["A2", 5000, "all", "ranged", 100],
        # ["A2", 7000, "all", "ranged", 100],
        # ["A1", 1250, "all", "fixed", 100],
        # ["A1", 1500, "all", "fixed", 100],
        # ["A1", 1750, "all", "fixed", 100],
        # ["A1", 2000, "all", "fixed", 100],
        # ["A2", 1250, "all", "ranged", 100],
        # ["A2", 1500, "all", "ranged", 100],
        # ["A2", 1750, "all", "ranged", 100],
        # ["A2", 2000, "all", "ranged", 100],
        ["C0", 1000, "all", "ranged", 100, "params_A2_2250.npy", "C"],
        ["C0", 2000, "all", "ranged", 100, "params_A2_2250.npy", "C"],
        ["C0", 3000, "all", "ranged", 100, "params_A2_2250.npy", "C"],
        ["C0", 4000, "all", "ranged", 100, "params_A2_2250.npy", "C"],
        ["C1", 1000, "all", "ranged", 100, "params_A2_2500.npy", "C"],
        ["C1", 2000, "all", "ranged", 100, "params_A2_2500.npy", "C"],
        ["C1", 3000, "all", "ranged", 100, "params_A2_2500.npy", "C"],
        ["C1", 4000, "all", "ranged", 100, "params_A2_2500.npy", "C"],
        ["D0", 1000, "all", "ranged", 100, "params_A2_2250.npy", "D"],
        ["D0", 2000, "all", "ranged", 100, "params_A2_2250.npy", "D"],
        ["D0", 3000, "all", "ranged", 100, "params_A2_2250.npy", "D"],
        ["D0", 4000, "all", "ranged", 100, "params_A2_2250.npy", "D"],
        ["D1", 1000, "all", "ranged", 100, "params_A2_2500.npy", "D"],
        ["D1", 2000, "all", "ranged", 100, "params_A2_2500.npy", "D"],
        ["D1", 3000, "all", "ranged", 100, "params_A2_2500.npy", "D"],
        ["D1", 4000, "all", "ranged", 100, "params_A2_2500.npy", "D"],
    ]
    dic = dict()
    for one_plan in plans:
        dic["generation"] = one_plan[1]
        dic["dataset"] = one_plan[2]
        dic["start"] = one_plan[3]
        dic["pop_size"] = one_plan[4]
        dic["params"] = one_plan[5]
        dic["diff_strategy"] = one_plan[6]

        one_slurm(
            "GA_{}_{}".format(one_plan[0], one_plan[1]),
            "test_nsga.py",
            dic)

def one_time_build_spatial():
    plans = [
        [3, 0.2, 1.0],
        [3, 0.2, 10.0],
        [3, 0.2, 100.0],
        [3, 0.2, 1000.0],
        [3, 0.5, 1.0],
        [3, 0.5, 10.0],
        [3, 0.5, 100.0],
        [3, 0.5, 1000.0],
        [3, 0.8, 1.0],
        [3, 0.8, 10.0],
        [3, 0.8, 100.0],
        [3, 0.8, 1000.0],
    ]
    dic = dict()
    for one_plan in plans:
        dic["n"] = one_plan[0]
        dic["threshold"] = one_plan[1]
        dic["diffusion_unit_rate"] = one_plan[2]
        one_slurm(
            "spatial_{}_{}".format(one_plan[1], one_plan[2]),
            "spatial_simulation.py",
            dic)

if __name__ == "__main__":
    one_time_build_spatial()
    pass
