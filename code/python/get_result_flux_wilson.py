import pandas as pd
from numba import njit
import sys
import os.path
import numpy as np
import math
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", ".."))
import statistics_python.src.statistics_observables as stat


def get_flux(data):
    x = data[['correlator', 'wilson', 'plaket']].to_numpy()

    # print(x)

    field, err = jackknife_var(x, field_electric)

    # print(field, err)

    return pd.Series([field, math.sqrt(err)], index=['field', 'err'])


def field_electric(x):
    a = x.mean(axis=0)
    return a[0] / a[1] - a[2]


@njit
def potential_numba(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        # if (x[1][i] == 0):
        #     print(i)
        fraction = x[0][i] / x[1][i] - x[2][i]
        if(fraction >= 0):
            y[i] = math.log(fraction)
        else:
            y[i] = 0
    return y


def jackknife(x, func):
    n = len(x)
    idx = np.arange(n)
    return sum(func(x[idx != i]) for i in range(n)) / float(n)


def jackknife_var(x, func):
    n = len(x)
    idx = np.arange(n)
    j_est = jackknife(x, func)
    return j_est, (n - 1) / (n + 0.0) * sum((func(x[idx != i]) - j_est)**2.0
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
    return sum(func(x[idx != i]) for i in range(n)) / float(n)


def test_transform(data):
    a = data.to_numpy()
    return a[0]


conf_type = "qc2dstag"
# conf_type = "su2_suzuki"
# conf_type = "SU2_dinam"
# conf_sizes = ["40^4", "32^4"]
conf_sizes = ["40^4"]
# conf_sizes = ["24^4"]
# conf_sizes = ["32^4"]
matrix_type = 'su2'

HYP_alpha = '1_1_0.5'
HYP_steps = 0
# APE_alpha1 = ['0.4', '0.45', '0.5', '0.6', '0.7']
APE_alpha1 = ['0.5']
# betas = ['2.6']
betas = ['/']

for beta in betas:
    for APE_alpha in APE_alpha1:
        for monopole in ['monopole', 'monopoless', 'su2']:
            # for monopole in ['monopole']:
            for conf_size in conf_sizes:
                if conf_size == '40^4':
                    conf_max = 1200
                    # conf_max = 1000
                    # mu1 = ['mu0.05', 'mu0.35', 'mu0.45']
                    # mu1 = ['0.05']
                    mu1 = ['mu0.00']
                    # chains = {"s0", "s1", "s2", "s3",
                    #           "s4", "s5", "s6", "s7", "s8"}
                    chains = {"/"}
                elif conf_size == '32^4':
                    conf_max = 2800
                    mu1 = ['mu0.00']
                    chains = {"/"}
                elif conf_size == '24^4':
                    conf_max = 100
                    mu1 = ['']
                    chains = {"/"}
                for mu in mu1:
                    # print(monopole, mu)
                    data = []

                    print(beta, monopole, conf_size, mu, APE_alpha)

                    for chain in chains:
                        for i in range(1, conf_max):
                            # file_path = f"../../data/flux_tube/qc2dstag/40^4/HYP_APE/mu{mu}/s{chain}/flux_dep_{i:04}"
                            file_path = f'../../data/flux_tube_wilson/{matrix_type}/{conf_type}/{conf_size}/{beta}/{mu}/'\
                                f'{matrix_type}-{monopole}/HYP{HYP_steps}_alpha={HYP_alpha}_APE_alpha={APE_alpha}/{chain}/electric_{i:04}'
                            # print(file_path)
                            if(os.path.isfile(file_path)):
                                data.append(pd.read_csv(file_path))
                                data[-1]["conf_num"] = i
                                # data[-1]["chain"] = f"s{chain}"

                    if len(data) == 0:
                        print("no data", beta, monopole,
                              conf_size, mu, APE_alpha)
                    elif len(data) != 0:
                        df = pd.concat(data)

                        # data_size = len(df['T'])

                        # data_d = np.zeros(data_size)
                        # for i in range(data_size / 4):
                        #     data_d[i * 4] = 0
                        #     data_d[i * 4 + 1] = 4
                        #     data_d[i * 4 + 2] = 0
                        #     data_d[i * 4 + 3] = 8

                        # print(df)

                        # df['d'] = pd.Series(data_d, index=df.index)

                        # print(df)

                        df2 = df.groupby(
                            ['smearing_step', 'T', 'R', 'd']).apply(get_flux).reset_index()

                        path_output = f"../../result/flux_tube_wilson/{matrix_type}/{conf_type}/{conf_size}/{beta}/{mu}/{matrix_type}-{monopole}"

                        try:
                            os.makedirs(path_output)
                        except:
                            pass

                        df2.to_csv(
                            f"{path_output}/flux_tube_wilson_HYP{HYP_steps}_alpha={HYP_alpha}_APE_alpha={APE_alpha}.csv", index=False)
