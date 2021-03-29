import pandas as pd
import os.path
import numpy as np
import math


def get_field(data, df1):
    x = data[['wilson-plaket-correlator',
              'wilson-loop', 'plaket']].to_numpy()

    field, err = jackknife_var(x, field_electric)

    new_row = {'field': field, 'err': math.sqrt(err)}

    df1 = df1.append(new_row, ignore_index=True)

    return df1


def field_electric(x):
    a = x.mean(axis=0)
    return a[0] / a[1] - a[2]


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


pd.set_option("display.precision", 12)

conf_type = "qc2dstag"


def get_field(data, arr_field, names):
    arr_field.append(data[names[0]].values[0] /
                     data[names[1]].values[0] - data[names[2]].values[0])


def field_electric(x):
    a = x.mean(axis=0)
    return a[0] / a[1] - a[2]


def jackknife(x, func):
    n = len(x)
    idx = np.arange(n)
    return sum(func(x[idx != i]) for i in range(n))/float(n)


def test_transform(data):
    a = data.to_numpy()
    return a[0]


chains = 1
mu = "0.40"
conf_size = "32^4"

data = []

for chain in range(chains):
    for i in range(1, 2758):
        # file_path = f"../../data/flux_tube/qc2dstag/40^4/HYP_APE/mu{mu}/s{chain}/flux_dep_{i:04}"
        file_path = f"../../data/flux_tube/qc2dstag/{conf_size}/HYP_APE/mu{mu}/flux_dep_{i:04}"
        if(os.path.isfile(file_path)):
            data.append(pd.read_csv(file_path))
            data[-1]["conf_num"] = i
            # data[-1]["chain"] = f"s{chain}"

df = pd.concat(data)

time_sizes = [8, 16]
space_sizes = [8, 16]

col_names = []

for T in time_sizes:
    for R in space_sizes:
        col_names.append(f"T={T}_R={R}_d=0_electric")
        col_names.append(f"T={T}_R={R}_d={R//2}_electric")
        col_names.append(f"T={T}_R={R}_wilson")
col_names.append("plaket")

df1 = df.groupby('smearing_step')[col_names].mean().reset_index()

# print(df1[["T=8_R=8_d=0_electric", "T=8_R=8_wilson"]].head())

col_names1 = []
for T in time_sizes:
    for R in space_sizes:
        col_names1.append([f"T={T}_R={R}_d=0_electric", f"T={T}_R={R}_wilson"])
        col_names1.append(
            [f"T={T}_R={R}_d={R//2}_electric", f"T={T}_R={R}_wilson"])

# x = df.loc[df["smearing_step"] == 0, ["T=8_R=8_d=4_electric", "T=8_R=8_wilson", "plaket"]
#            ].to_numpy()

# print(x[2].mean())
# print(jackknife(x, field_electric))

# df.groupby('smearing_step')[
#     "T=8_R=8_d=0_electric"].apply(test_transform)
# print(df)

names = ["T=16_R=16_d=8_electric", "T=16_R=16_wilson", "plaket"]

arr_field = []
df1.groupby('smearing_step')[col_names].apply(get_field, arr_field, names)
print(arr_field)
df1.insert(len(df1.columns), 'field', arr_field)
df2 = df1[['smearing_step', 'field']]
print(df2)
df2.to_csv(
    f"../../result/flux_tube/qc2dstag/{conf_size}/HYP_APE/flux_tube_mu{mu}.csv")
