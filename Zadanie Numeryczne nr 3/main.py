import numpy as np
import matplotlib as plt

N = 5
x = [0 for i in range(N)]
mat = [[0 for i in range(N)] for j in range(N)] 

for i in range(N):
    x[i] = i+1

    mat[i][i] = 1.2
    if (i+1 < N):
        mat[i+1][i] = 0.2 # set 0.2 under diag
        mat[i][i+1] = (0.1)/(i+1) # set 0.1/(i+1) under diag
    if (i+2 < N):
        mat[i][i+2] = (0.15)/(i+1)**2 # set 0.15/(i+1)**2 under diag


def calcU(i, mat, matL, matU):
    u = mat[i][i] - matL[i][i-1] * matU[i-1][i]
    matU[i][i] = u

def calcL1(i, mat, matL, matU):
    l1 = (mat[i+1][i]) / matU[i][i]
    matL[i+1][i] = l1

def calcU1(i, mat, matL, matU):
    u1 = mat[i][i+1] - matL[i][i+1]*matU[i-1][i+1]
    matU[i+1][i] = u1

def calcU2(i, mat, matL, matU):
    u2 = mat[i][i+2]
    matU[i+2][i] = u2

print(np.matrix(mat))