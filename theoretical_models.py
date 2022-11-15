from math import log as ln
from math import sqrt, pi, exp
# from scipy import integrate
# import numpy as np
import mpmath
mpmath.mp.dps = 100

EPSILON = 10**-9
INFINITY = 10**7


def Queue_TAS(T, A, N):
    return (N+1) / 2


def Queue_WAS(T, A, N):
    return 32 / 33 * N


def RW_TAS(T, A, N):
    return T / N * ln(A / (A-N))


def RW_WAS(T, A, N):
    S = 0
    for k in range(INFINITY):
        Si = 1 - (1 - ((T-A+N)/T)**k)**32
        S += Si
        # if k >= 1000 and Si < EPSILON:
        #     break
    return S


def RWBM_TAS(T, A, N):
    S = 0
    for k in range(N-1):
        Si = 1 / (1 - ((T-A+k)/T)**32) / N
        S += Si
        # if k >= 1000 and Si < EPSILON:
        #     break
    return S


def RWBM_WAS(T, A, N):
    S = 0
    for k in range(INFINITY):
        Si = 1 - (1 - ((T-A+N)/T)**(32*k))**32
        S += Si
        # if k >= 1000 and Si < EPSILON:
        #     break
    return S


def CoRWBM_WAS(T, A, N):
    mu = 1024 * (A-N) / T
    sigma = 1024 * (A-N)/T * (T-A+N)/T
    EB = mpmath.quad(
        lambda t: 32 / (sigma * sqrt(2*pi*t)) *
        exp(-(32-mu*t)**2 / (2*(sigma**2)*t)),
        [0, mpmath.inf]
    )
    return max(1, EB)
