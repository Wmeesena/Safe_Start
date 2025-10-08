import numpy as np
import torch

# Helpers for pretty printing
def pct(x): 
    return f"{100.0 * x:.2f}%"

def pct_ci(ci):
    lo, hi = ci
    return f"[{100.0*lo:.2f}%, {100.0*hi:.2f}%]"