import pandas as pd
import os.path
import numpy as np
import math
import sys, getopt


def get_field(data, df1, names):
    x = data[f"wilson_{name}"].to_numpy()

    field, err = jackknife_var(x, field_wilson)

    new_row = {f"{name}_wilson": field, f"{name}_err": math.sqrt(err)}

    df1 = df1.append(new_row, ignore_index=True)

    return df1


def field_wilson(x):
    a = x.mean(axis=0)
    return a


def jackknife(x, func):
    n = len(x)
    idx = np.arange(n)
    return sum(func(x[idx != i]) for i in range(n))/float(n)


def jackknife_var(x, func):
    n = len(x)
    idx = np.arange(n)
    j_est = jackknife(x, func)
    return j_est, (n-1)/(n + 0.0) * sum((func(x[idx != i]) - j_est)**2.0
                                        for i in range(n))


longoptions = ["mu=", "conf_type=", "chains=", "conf_size=", "conf_num=", "conf_path=", "smearing=", "alpha_APE=", "monopole="]

opts, args = getopt.getopt(sys.argv[1:], "", longoptions)

for opt, arg in opts:
    if opt == '--mu':
        mu = arg
    elif opt == '--conf_type':
        conf_type = arg
    # elif opt == "--chains":
    #     chains = arg
    elif opt == "--conf_size":
        conf_size = arg
    elif opt == "--conf_num":
        conf_num = int(arg)
    elif opt == "--conf_path":
        conf_path = arg
    elif opt == "--smearing":
        smearing = arg
    elif opt == "--alpha_APE":
        alpha_APE = arg
    elif opt == "--monopole":
        monopole = arg

# chains = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
chains = [""]
data = []
# monopole = 'monopoless'
# monopole = ''
# monopole = ''

for chain in chains:
    for i in range(1, conf_num):
        file_path = f"../../data/wilson_loop/{monopole}/{conf_type}/{conf_size}/{smearing}/mu{mu}/{chain}/wilson_loop_{alpha_APE}_{i:04}"
        if(os.path.isfile(file_path)):
            data.append(pd.read_csv(file_path))
            # data[-1]["conf_num"] = i
            # data[-1]["chain"] = f"s{chain}"

df = pd.concat(data)

# df_test = df[np.isnan(df['wilson_loop'])]

# print(df_test)

time_sizes = [8, 16]
space_sizes = [8, 16]

col_names = []

for T in time_sizes:
    for R in space_sizes:
        col_names.append(f"T={T}_R={R}")

df2 = []

for name in col_names:
    df1 = pd.DataFrame(columns=[f"{name}_wilson", f"{name}_err"])
    df2.append(df.groupby(['smearing_step']).apply(get_field, df1, name).reset_index() [['smearing_step', f"{name}_wilson", f"{name}_err"]])

df2 = pd.concat(df2, axis = 1)

df2 = df2.loc[:,~df2.columns.duplicated()]

# print(df2)

path_output = f"../../result/wilson_loop/{monopole}/{conf_type}/{conf_size}/{smearing}"

try:
    os.makedirs(path_output)
except:
    pass

df2.to_csv(f"{path_output}/wilson_loop_mu{mu}_alpha_APE={alpha_APE}.csv")
