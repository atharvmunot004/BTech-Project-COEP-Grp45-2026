"""
Extreme Value Theory (EVT) - Peaks Over Threshold (POT) for VaR and CVaR evaluation.

This module implements EVT-POT methodology for estimating extreme risk measures
using the Generalized Pareto Distribution (GPD) fitted to exceedances over a threshold.
"""

from .main import evaluate_evt_pot_var_cvar

__all__ = ['evaluate_evt_pot_var_cvar']

