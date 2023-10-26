import numpy as np
import matplotlib as plt
import time

def initMatrix(N, mat):
    for i in range(N):
        mat[i][i] = 1.2
        if (i+1 < N):
            mat[i+1][i] = 0.2 # set 0.2 under diag
            mat[i][i+1] = (0.1)/(i+1) # set 0.1/(i+1) under diag
        if (i+2 < N):
            mat[i][i+2] = (0.15)/(i+1)**2 # set 0.15/(i+1)**2 under diag

# todo: sprawdz jak sie liczylo wyzacznik L, U skoro nie sa diagonalne

def measurePerformance(N):
    x = [0 for i in range(N)]
    for i in range(N):
        x[i] = i+1
    
    mat = [[0 for i in range(N)] for j in range(N)] 

    arrU = [0 for i in range(N)] 
    arrL1 = [0 for i in range(N)] 
    arrU1 = [0 for i in range(N)] 
    arrU2 = [0 for i in range(N)] 

    initMatrix(N, mat)
    start = time.time() * 1000

    for i in range(N):
        # calc U
        if (i-1 <= N):
            arrU[i] = mat[i][i]
        else:
            arrU[i] = mat[i][i] - arrL1[i-1] * arrU1[i]
        
        # calc L1
        if (i+1 < N): 
            arrL1[i] = (mat[i+1][i]) / arrU[i]

        # calc U1
        if (i+1 < N and i-1 > -1):
            arrU1[i] = mat[i][i+1] - arrL1[i]*arrU2[i]

        # calc U2
        if (i+2 < N):
            arrU2[i] = mat[i][i+2]

    end = time.time() * 1000
    delta = end - start


    # create L matrix
    matL = [[0 for i in range(N)] for j in range(N)] 
    for i in range(N):
        matL[i][i] = 1
        if (i+1 < N):
            matL[i+1][i] = arrL1[i]

    # create U matrix
    matU = [[0 for i in range(N)] for j in range(N)] 
    for i in range(N):
        matU[i][i] = arrU[i]
        if (i+1 < N):
            matU[i][i+1] = arrU1[i]
        if (i+2 < N):
            matU[i][i+2] = arrU2[i]

    print("Og:")
    print(np.matrix(mat))
    print("My:")
    print(np.matmul(np.matrix(matL), np.matrix(matU)))
    print("N: " + str(N) + " Time in ms: " + str(delta))

measurePerformance(5)

