import numpy as np
import matplotlib.pyplot as plt
import math
import time

def vectorNorm(A):
    sum = 0
    for i in range(len(A)):
        sum += A[i]*A[i]
    return math.sqrt(sum)

def scalarMulVec(scalar, A):
    copy = A.copy()
    for i in range(len(A)):
        copy[i] = copy[i] * scalar
    return copy


def eigenvaluePowerMethod(A):
    startVec = [1 for _ in range(len(A))]

    while(True):
        norm = vectorNorm(startVec)
        normedVec = scalarMulVec(norm, startVec)

        # pomnóż A * normedVec i rozwiaz uklad A * normedVec = x_i
        # x_i unormuj i to bedzie normedVec w nastepnej iteracji
        # wykonuj dopóki y_m - y_m-1 < epsilon


def solve():
    return 1






