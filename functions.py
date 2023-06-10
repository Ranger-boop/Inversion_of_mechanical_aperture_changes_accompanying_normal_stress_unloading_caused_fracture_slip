import numpy as np
import pandas as pd


def check_0(vlc: pd.Series) -> None:
    """
    Check how many zero(s) are in the vlocities.

    Parameters
    ----------
    vlc : ndarray
        The ndarray of velocities need to be checked.
    output : text.
        The text displaying the number of zero(s).
    """
    num_0 = 0
    for i in vlc.index:
        if vlc[i] == 0:
            num_0 += 1
    print(f"{num_0} 0s are in the velocity list.")


def aperture_slip_disp(
    b_0: float, u_end: np.ndarray, u0_ini: float, dil_ang: float
) -> np.ndarray:
    """
    Compute the fracture aperture dependent on slip displacement when the
    change of shear velocity is not considered.

    Parameters
    ----------
    b_0 : float
        Initial aperture at the onset of reactivation (the beginning of the
        first velocity step).
    u_end : ndarray
        Slip displacement after the reactivation of fracture at the end of
        each velocity step.
    u0_ini : float
        Slip displacement at the onset of reactivation.
    dil_ang : float
        Dilation angle of the fault, the unit here is degree.
    output : ndarray
        Fracture aperture dependent on slip displacement with the change of
        shear velocity not considered.
    """
    delta_u = u_end - u0_ini
    b_slip = b_0 + delta_u * np.tan(dil_ang * np.pi / 180)

    return b_slip


def dil_para(
    dil_fact: float, u_end: np.ndarray, v: np.ndarray, D_c: float, dt_acq: float = 1
) -> np.ndarray:
    """
    Compute the dilation parameter influenced by shear velocity.

    Parameters
    ----------
    dil_fact : float
        The dilation factor that controls the dilation of fault.
    u_end : ndarray
        Slip displacement at the end of each velocity step.
    v : ndarray
        Slip velocity at each velocity step.
    D_c : float
        Characteristic distance of the fault.
    dt_acq : float, optional
        Acquisition interval for recorded data, the default value is 1s.
    output : ndarray
        The computed dilation parameters at each velocity step for the last
        velocity step in v.
    """
    d_phi = np.zeros(len(v))
    d_phi[0] = 0

    if len(v) > 1:
        v_b = v[:-1]
        v_f = v[1:]

        dt = np.zeros(len(v))
        # for i in range(1, len(dt) - 1):
        #     dt[i] = (u_end[i + 1] - u_end[i]) / v[i]
        dt = dt_acq + (u_end[-1] - u_end) / v
        dt[0] = 0
        dt[-1] = dt_acq

        d_phi[1:] = -dil_fact * np.log(
            v_b / v_f * (1 + (v_f / v_b - 1) * np.exp(-v_f * dt[1:] / D_c))
        )

    return d_phi


def b_mod_n(b_slip_n: float, d_phi_1dim: np.ndarray) -> float:
    """
    Compute the aperture influenced by both displacement and velocity of the
    fracture at the velocity step of b_slip_n.

    Parameters
    ----------
    b_slip_n : float
        The fracture aperture dependent on slip displacement at the n_th
        velocity step.
    d_phi_1dim : ndarray
        Dilation parameters at each velocity step before the step of b_slip_n.
    output : float
        Aperture at the velocity step of b_slip_n.
    """
    ratio_bslip_cumprod = 1
    for i in d_phi_1dim:
        ratio_bslip_cumprod *= 1 + i

    return b_slip_n * ratio_bslip_cumprod


def aperture_shear_dil(b_slip: np.ndarray, d_phi_2dim: list[np.ndarray]) -> np.ndarray:
    """
    Compute fracture aperture influenced by slip displacement and velocity at
    each velocity step in the sequence of b_slip.

    Parameters
    ----------
    b_slip : ndarray
        The aperture influenced only by the slip displacement of fracture,
        should be a sequence to be computed.
    d_phi_2dim : list
        Sequences of dilation parameter for each velocity step in b_slip.
        The lengths of these sequences should be different and increasing.
        The items in d_phi_2dim should be ndarray.
    output : ndarray
        Aperture influenced by both slip displacement and velocity of the
        fracture at each velocity step in b_slip.
    """
    b_mod = np.zeros(len(b_slip))
    for i in range(len(b_slip)):
        b_mod[i] = b_mod_n(b_slip[i], d_phi_2dim[i])

    return b_mod


def rmse(b_mod: np.ndarray, b_exp: np.ndarray) -> float:
    """
    Compute and return the value of Root Mean Square Error (RMSE)
    between the calculated and measured aperture.

    Parameters
    ----------
    b_mod : ndarray
        The calculated aperture containing shear dilation.
    b_exp : ndarray
        The measured aperture containing shear dilation.
    output : float
        The computed RMSE of the two sets of data.
    """
    return np.sqrt(np.sum((b_mod - b_exp) ** 2) / len(b_mod))  # type: ignore
