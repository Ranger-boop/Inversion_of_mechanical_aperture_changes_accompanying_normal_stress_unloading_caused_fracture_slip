import numpy as np
import pandas as pd
from sko.GA import GA
from functions import *


# Import data to be utilized for inversion
# Replace the value of dil_file with the excel data containing
# slip displacement, slip velocity, normal displacement, and
# normal stress of the direct shear test.
# Replace the value of skip_rows with the number of rows before
# the header of the data table.
dil_file = "..."
skip_rows = 1
data_dil = pd.read_excel(dil_file, skiprows=skip_rows)


# Compute the slip velocity of the fracture
time_interval = 1  # The time interval of the acquisition system
data_dil["velocity[mm/s]"] = data_dil["Slip displacement (mm)"].diff() / time_interval
data_dil.loc[data_dil[data_dil["velocity[mm/s]"] < 0].index, "velocity[mm/s]"] = 0


# Adjust unreasonable slip displacement
for i in data_dil.index[1:]:
    if (
        data_dil.loc[i, "Slip displacement (mm)"]
        - data_dil.loc[i - 1, "Slip displacement (mm)"]
        < 0
    ):
        data_dil.loc[i, "Slip displacement (mm)"] = data_dil.loc[
            i - 1, "Slip displacement (mm)"
        ]

# Set the time duration of pre-slip segment
dur_n_stiff = [252, 376]
# Set the time duration of stable sliding segment
dur_sss = [388, 694]
# Extract slip velocity in stable sliding segment
vlc = data_dil[(data_dil.index >= dur_sss[0]) & (data_dil.index <= dur_sss[1])][
    "velocity[mm/s]"
].copy()


# Adjust the slip velocity to inverse correctly
for i in vlc.index:
    if vlc[i] == 0:
        vlc[i] = (vlc[i - 1] + vlc[i + 1]) / 2


# Compute normal stiffness of the testing system
coeff = np.polyfit(
    data_dil[(data_dil.index >= dur_n_stiff[0]) & (data_dil.index <= dur_n_stiff[1])][
        "Normal stress (Mpa)"
    ],
    data_dil[(data_dil.index >= dur_n_stiff[0]) & (data_dil.index <= dur_n_stiff[1])][
        "Average normal displacement (mm)"
    ],
    1,
)
norm_stiff = 1 / coeff[0]


# Compute aperture during shear dilation
# Maximum fracture aperture computed as RMS asperity height, the unit is mm
b_max = 1.340498
b_ini = (
    b_max - data_dil["Normal stress (Mpa)"][0] / norm_stiff
)  # Aperture at the onset of test
data_dil["aperture[mm]"] = b_ini - (
    data_dil["Average normal displacement (mm)"]
    - data_dil["Average normal displacement (mm)"][0]
)  # Aperture during the whole test

deform_unload = (
    data_dil.loc[vlc.index, "Normal stress (Mpa)"]
    - data_dil.loc[vlc.index[0], "Normal stress (Mpa)"]
) / norm_stiff  # Unloading caused aperture change

# Aperture without unloading caused aperture change
b_exp = apert_no_unload = data_dil.loc[vlc.index, "aperture[mm]"] + deform_unload


# Set parameters for inversion
# Arguments to compute b_slip (displacement-dependent aperture)
b_0 = b_exp.values[0]
u0_ini = data_dil.loc[vlc.index, "Slip displacement (mm)"].values[0]
u_end = data_dil.loc[vlc.index, "Slip displacement (mm)"].values[1:]

# Arguments to compute dilation parameters
v = vlc.values[1:]


def fun_obj(paras):
    """
    The objective function to be optimized with Genetic Algorithm.

    Parameters
    ----------
    paras :
        The set of constrained parameters to be computed.
    output :
        RMSE (root mean square error) between computed and measured aperture.
    """
    # Receive dilation factor, characteristic slip distance, and dilation angle
    # with their upper and lower bounds constrained
    dil_fact, D_c, dil_ang = paras

    # Displacement-dependent aperture
    b_slip: np.ndarray = aperture_slip_disp(b_0, u_end, u0_ini, dil_ang)

    # Dilation parameters
    d_phi_2dim: list[np.ndarray] = [0] * len(b_slip)  # type: ignore
    for i in range(len(b_slip)):
        d_phi_2dim[i] = dil_para(dil_fact, u_end[: i + 1], v[: i + 1], D_c, dt_acq=1)

    # Modeled aperture
    b_mod: np.ndarray = aperture_shear_dil(b_slip, d_phi_2dim)

    # Root mean square error
    rmse_obj = rmse(b_mod, b_exp.values[1:])

    return rmse_obj


ga = GA(
    func=fun_obj,
    n_dim=3,
    size_pop=200,
    max_iter=10000,
    # Lower bounds of dil_fact (dilation factor), D_c (characteristic distance),
    # and dil_ang (dilation angle)
    lb=[0, 1e-5, 0],
    # Upper bounds of dil_fact (dilation factor), D_c (characteristic distance),
    # and dil_ang (dilation angle)
    ub=[0.5, 2, 20],
    #     precision=1e-7,
)
best_x, best_y = ga.run()


# Output the best-fit parameters
print(
    f"Best-fit dilation factor: {best_x[0]},\n"
    + f"Best-fit D_c: {best_x[1]},\n"
    + f"Best-fit dilation angle: {best_x[2]}"
)
print(f"Lowest value of objective function: {best_y[0]}")
