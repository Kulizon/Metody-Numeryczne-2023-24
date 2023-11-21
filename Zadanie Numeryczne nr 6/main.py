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

def maxEigenvalueNumpy(A):
    eigenvalues = np.linalg.eigvals(A)
    maxEigenvalue = np.max(np.abs(eigenvalues)) 
    return maxEigenvalue

def eigenvaluePowerMethod():
    # arrUnderDiag = [0, 1, 2, 3]
    # arrDiag = [8, 7, 6, 5]
    # arrOverDiag = [1, 2, 3, 0]

    A = [[8, 1, 0, 0],
         [1, 7, 2, 0],
         [0, 2, 6, 3],
         [0, 0, 3, 5]]

    curVec = [1 for _ in range(len(A))]
    resultVecs = [curVec]
    resultLambdas = [1]

    while(True):
        curVec = matMulVec(A, curVec)
        lamb, normedVec = normalize(curVec)
        curVec = normedVec

        resultVecs.append(normedVec)
        resultLambdas.append(lamb)
        
        # pomnóż A * normedVec i rozwiaz uklad A * normedVec = x_i
        # x_i unormuj i to bedzie normedVec w nastepnej iteracji
        # wykonuj dopóki y_m - y_m-1 < epsilon

        # jednak pewnie bedzie wolno zbiegac wiec nalezy przesunac A - p * 1

        diff = abs(vecNorm(resultVecs[len(resultVecs) - 1]) - vecNorm(resultVecs[len(resultVecs) - 2]))
        if (diff < 0.00000001):
            break

    print(resultVecs[len(resultVecs) - 1])
    print(len(resultLambdas))
    print(resultLambdas[len(resultLambdas) - 1], maxEigenvalueNumpy(A))


eigenvaluePowerMethod()

def solve():
    return 1






