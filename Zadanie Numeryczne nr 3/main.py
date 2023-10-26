import numpy as np
import matplotlib as plt
import time

def initMatrixes(N, mat, matL, matU):
    for i in range(N):
        matL[i][i] = 1

        mat[i][i] = 1.2
        if (i+1 < N):
            mat[i+1][i] = 0.2 # set 0.2 under diag
            mat[i][i+1] = (0.1)/(i+1) # set 0.1/(i+1) under diag
        if (i+2 < N):
            mat[i][i+2] = (0.15)/(i+1)**2 # set 0.15/(i+1)**2 under diag


def calcU(i, N, mat, matL, matU):
    if (i-1 <= N):
        matU[i][i] = mat[i][i]
        return

    u = mat[i][i] - matL[i][i-1] * matU[i-1][i]
    matU[i][i] = u

def calcL1(i, N, mat, matL, matU):
    if (i+1 >= N): 
        return


    l1 = (mat[i+1][i]) / matU[i][i]
    matL[i+1][i] = l1

def calcU1(i, N, mat, matL, matU):
    if (i+1 >= N or i-1 <= -1):
        return # todo: maybe split these cases into 2 ant work in i-1 <= -1

    u1 = mat[i][i+1] - matL[i][i+1]*matU[i-1][i+1]
    matU[i][i+1] = u1

def calcU2(i, N, mat, matL, matU):
    if (i+2 >= N):
        return

    u2 = mat[i][i+2]
    matU[i][i+2] = u2


def printMatrixes(mat, matL, matU):
    print(np.matrix(mat))
    print(np.matrix(matL))
    print(np.matrix(matU))
    print()

def performCalculations(N, mat, matL, matU):
    for i in range(N):
        calcU(i, N, mat, matL, matU)
        calcL1(i, N, mat, matL, matU)
        calcU1(i, N, mat, matL, matU)
        calcU2(i, N, mat, matL, matU)

        #todo: move this prob

# todo: sprawdz jak sie liczylo wyzacznik L, U skoro nie sa diagonalne

# todo: pass arrays as pointers or sth so coppying doesnt worsen the performance

def measurePerformance(N):
    x = [0 for i in range(N)]
    for i in range(N):
        x[i] = i+1
    
    mat = [[0 for i in range(N)] for j in range(N)] 
    matL = [[0 for i in range(N)] for j in range(N)] 
    matU = [[0 for i in range(N)] for j in range(N)] 
    initMatrixes(N, mat, matL, matU)
    start = time.time() * 1000

    performCalculations(N, mat, matL, matU)

    end = time.time() * 1000
    delta = start - end

    # print("Og:")
    # print(np.matrix(mat))
    # print("My:")
    # print(np.matmul(np.matrix(matL), np.matrix(matU)))
    print("N: " + str(N) + " Time in ms: " + str(delta))


measurePerformance(1240)

