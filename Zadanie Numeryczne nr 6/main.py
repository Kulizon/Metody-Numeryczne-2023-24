import numpy as np
import matplotlib.pyplot as plt
import math
import time

def vecNorm(A):
    sum = 0
    for i in range(len(A)):
        sum += A[i]*A[i]
    return math.sqrt(sum)

def normalize(vec):
    copy = vec.copy()
    maxEl = max(vec)
    for i in range(len(vec)):
        copy[i] = copy[i] / maxEl
    return maxEl, copy

def matMulVec(mat, vec):
    result = []
    for i in range(len(mat)):
        element = 0
        for j in range(len(vec)):
            element += mat[i][j] * vec[j]

        result.append(element)
    return result

def eigenvaluePowerMethod():
    # arrUnderDiag = [0, 1, 2, 3]
    # arrDiag = [8, 7, 6, 5]
    # arrOverDiag = [1, 2, 3, 0]

    A = [[4, 1, 0],
         [0, 2, 1],
         [0, 0, -1]]

    curVec = [1 for _ in range(len(A))]
    result = [curVec]

    i = 0
    while(True):
    
        
        curVec = matMulVec(A, curVec)
        lamb, normedVec = normalize(curVec)
        curVec = normedVec

        print(lamb, curVec, normedVec)

        result.append(normedVec)
        
        # pomnóż A * normedVec i rozwiaz uklad A * normedVec = x_i
        # x_i unormuj i to bedzie normedVec w nastepnej iteracji
        # wykonuj dopóki y_m - y_m-1 < epsilon

        # jednak pewnie bedzie wolno zbiegac wiec nalezy przesunac A - p * 1

        diff = abs(vecNorm(result[len(result) - 1]) - vecNorm(result[len(result) - 2]))
        i += 1
        if (diff < 0.0001 or i > 10):
            break

    print(result[len(result) - 1])


eigenvaluePowerMethod()

def solve():
    return 1






