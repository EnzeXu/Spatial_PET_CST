from ode_truth import get_now_string


def pass_omega():
    with open("figure_spatial/record_20230202.csv", "a") as f:
        f.write("----------,pass,{}\n".format(get_now_string()))


if __name__ == "__main__":
    pass_omega()
