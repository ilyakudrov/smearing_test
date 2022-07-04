from numba import njit
import sys
import math
import time
import numpy as np
import os.path
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", ".."))
import statistics_python.src.statistics_observables as stat


def get_field(data, df1, df, time_size_max):

    time_size = data["T"].iloc[0]
    space_size = data["r/a"].iloc[0]
    smearing_step = data["smearing_step"].iloc[0]

    # print(smearing_step, time_size,  space_size)

    if time_size < time_size_max:

        x1 = data['wilson_loop'].to_numpy()

        x2 = df[(df['smearing_step'] == smearing_step) & (df["T"] == time_size + 1) & (df["r/a"]
                                                                                       == space_size)]['wilson_loop'].to_numpy()

        x3 = np.vstack((x1, x2))

        # field, err = stat.jackknife_var(x3, potential)
        field, err = stat.jackknife_var_numba(x3, potential_numba)

        # field1 = average(x3)

        new_row = {'aV(r)': field, 'err': err}

        df1 = df1.append(new_row, ignore_index=True)

        return df1


@njit
def potential_numba(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        # if (x[1][i] == 0):
        #     print(i)
        fraction = x[0][i] / x[1][i]
        if(fraction >= 0):
            y[i] = math.log(fraction)
        else:
            y[i] = 0
    return y


def potential(x):
    a = np.mean(x, axis=1)
    fraction = a[0] / a[1]
    if(fraction >= 0):
        return math.log(fraction)
    else:
        return 0


# conf_type = "qc2dstag"
conf_type = "su2_suzuki"
# conf_type = "SU2_dinam"
# conf_sizes = ["40^4", "32^4"]
# conf_sizes = ["40^4"]
conf_sizes = ["24^4"]
# conf_sizes = ["32^4"]
theory_type = 'su2'
betas = ['beta2.4', 'beta2.5', 'beta2.6']

# HYP_alpha = '1_0.5_0.5'
# HYP_alpha = '0.75_0.6_0.3'
HYP_alpha = '1_0.5_0.5'
HYP_steps = 1
# APE_alpha1 = ['0.5', '0.55', '0.6', '0.65', '0.7']
APE_alpha1 = ['0.5']

# for monopole in ['/', 'monopoless']:
# for monopole in ['monopoless']:
for beta in betas:
    for APE_alpha in APE_alpha1:
        for monopole in ['/', 'monopole', 'monopoless']:
            if monopole == '/':
                monopole1 = theory_type
            for conf_size in conf_sizes:
                if conf_size == '40^4':
                    conf_max = 1200
                    # conf_max = 1000
                    # mu1 = ['0.05', '0.35', '0.45']
                    mu1 = ['0.45']
                    # mu1 = ['0.00']
                    chains = {"s0", "s1", "s2", "s3",
                              "s4", "s5", "s6", "s7", "s8"}
                    # chains = {"/"}
                elif conf_size == '32^4':
                    conf_max = 2800
                    mu1 = ['0.00']
                    chains = {"/"}
                elif conf_size == '24^4':
                    conf_max = 100
                    mu1 = ['']
                    chains = {"/"}
                for mu in mu1:
                    # print(monopole, mu)
                    data = []
                    for chain in chains:
                        for i in range(0, conf_max):
                            # file_path = f"../../data/wilson_loop/{axis}/{monopole}/qc2dstag/{conf_size}/mu{mu}/{chain}/wilson_loop_{i:04}"
                            file_path = f"../../data/potential/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{monopole1}/{chain}/wilson_loops_{i:04}"
                            # print(file_path)

                            if(os.path.isfile(file_path)):
                                # print(file_path)
                                # data.append(pd.read_csv(file_path, header=0,
                                #                         names=["smearing_step", "T", "r/a", "wilson_loop"]))
                                # try:
                                data.append(pd.read_csv(file_path, header=0,
                                                        names=["smearing_step", "T", "r/a", "wilson_loop"]))
                                # except:
                                #     pass
                                # print("Deleting", file_path)
                                # os.remove(file_path)
                                data[-1]["conf_num"] = i
                    if len(data) == 0:
                        print("no data", conf_size, mu, APE_alpha)
                    elif len(data) != 0:
                        # try:
                        df = pd.concat(data)

                        # df = df[df['T'] <= 16]
                        # df = df[df['r/a'] <= 16]

                        # df_test = df[np.isnan(df['wilson_loop']) ]

                        # print(df_test)

                        # print(df)

                        # wilson = df[['wilson_loop']].to_numpy()
                        # conf_num = df[['conf_num']].to_numpy()

                        # for i in range(len(wilson)):
                        #     if(math.isnan(wilson[i])):
                        #         print(conf_num[i])

                        df1 = pd.DataFrame(columns=["aV(r)", "err"])

                        time_size_max = df["T"].max()

                        start = time.time()

                        df1 = df.groupby(['smearing_step', 'T', 'r/a']).apply(get_field, df1,
                                                                              df, time_size_max).reset_index()

                        end = time.time()
                        print("execution time = %s" % (end - start))

                        df1 = df1[['smearing_step',
                                   'T', 'r/a', 'aV(r)', 'err']]

                        path_output = f"../../result/potential/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}"
                        # path_output = f"../../result/potential_spatial/{test}/{axis}/{monopole}/qc2dstag/{conf_size}"
                        # path_output = f"../../result/potential/{test}/{axis}/{monopole}/su2_dynam/{conf_size}/{smearing}"

                        try:
                            os.makedirs(path_output)
                        except:
                            pass

                        # df1.to_csv(f"{path_output}/potential_spatial_mu={mu}.csv", index=False)
                        df1.to_csv(
                            f"{path_output}/potential_{monopole1}_HYP{HYP_steps}_alpha={HYP_alpha}_APE_alpha={APE_alpha}.csv", index=False)
                        # except:
                        #     pass
