import os
import sys
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import torch
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
import multiprocessing as mp
from tqdm import tqdm

class Logger:
    """
    A logger class that redirects the STDOUT and STDERR to a specified output file while
    preserving the output on screen. This is useful for logging terminal output to a file
    for later analysis while still seeing the output in real-time during execution.

    Parameters
    ----------
    logfile : str
        The file path of which the standard output and standard error should be logged.

    Attributes
    ----------
    terminal : :code:`io.TextIOWrapper` object
        The original standard output object, typically :code:`sys.stdout`.
    log : :code:`io.TextIOWrapper` object
        File object used to log the output in append mode.
    """

    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        """
        Writes a message to the terminal and to the log file.

        Parameters
        ----------
        message : str
            The message to be written to STDOUT and the log file.
        """
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure the message is written immediately

    def flush(self):
        """
        This method is needed for Python 3 compatibility. This handles the flush command by doing nothing.
        Some extra behaviors may be specified here.
        """
        # self.terminal.log()
        pass

def format_time(t):
    """
    Converts time in seconds to a more readable format.

    Parameters
    ----------
    t : float
        The time in seconds.

    Returns
    -------
    t_str : str
        A string representing the time duration in a format of "X hour(s) Y minute(s) Z second(s)", adjusting the units
        as necessary based on the input duration, e.g., 1 hour(s) 0 minute(s) 0 second(s) for 3600 seconds and
        15 minute(s) 30 second(s) for 930 seconds.
    """
    hh_mm_ss = str(datetime.timedelta(seconds=t)).split(":")

    if "day" in hh_mm_ss[0]:
        # hh_mm_ss[0] will contain "day" and cannot be converted to float
        hh, mm, ss = hh_mm_ss[0], float(hh_mm_ss[1]), float(hh_mm_ss[2])
        t_str = f"{hh} hour(s) {mm:.0f} minute(s) {ss:.0f} second(s)"
    else:
        hh, mm, ss = float(hh_mm_ss[0]), float(hh_mm_ss[1]), float(hh_mm_ss[2])
        if hh == 0:
            if mm == 0:
                t_str = f"{ss:.1f} second(s)"
            else:
                t_str = f"{mm:.0f} minute(s) {ss:.0f} second(s)"
        else:
            t_str = f"{hh:.0f} hour(s) {mm:.0f} minute(s) {ss:.0f} second(s)"

    return t_str


