"""General use functions.
"""
import os
import time
import argparse
import numpy as np
from scipy.optimize import nnls

################################################################################

def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > .04 or i%(int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep*i - elapsed)

################################################################################

def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")


def randomize_target_point():
    # Generate random values for each range
    value_1 = np.random.uniform(-1, 1)
    value_2 = np.random.uniform(-1, 1)
    value_3 = np.random.uniform(0, 1)

    # Create a numpy array with the three values
    return np.array([value_1, value_2, value_3])


def randomize_initial_positions(number_of_drones):
    """
    Randomly updates self.INIT_XYZS for each drone:
    - X and Y in range [-1, 1]
    - Z in range [0, 1]
    """
    return np.vstack([
        np.random.uniform(-1, 1, size=number_of_drones),      # Random X between -1 and 1
        np.random.uniform(-1, 1, size=number_of_drones),      # Random Y between -1 and 1
        np.random.uniform(0, 1, size=number_of_drones)    # Random Z between 0.125 and 1
    ]).transpose().reshape(number_of_drones, 3)


def get_norm_path(path: str):
    ret_path = os.path.join(
        os.path.dirname(__file__),
        path
    )

    return os.path.normpath(ret_path)
