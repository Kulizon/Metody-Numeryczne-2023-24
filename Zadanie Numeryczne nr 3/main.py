import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import time

def initMatrix(N, mat):
    for i in range(N):
        mat[i][i] = 1.2
        if (i+1 < N):
            mat[i+1][i] = 0.2 # set 0.2 under diag
            mat[i][i+1] = (0.1)/(i+1) # set 0.1/(i+1) under diag
        if (i+2 < N):
            mat[i][i+2] = (0.15)/(i+1)**2 # set 0.15/(i+1)**2 under diag

def debug(N, arrU, arrU1, arrU2, arrL1):
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

    # create original matrix by multiplying L and U
    LU = np.matmul(np.matrix(matL), np.matrix(matU))

    print("Matrix L")
    print(np.matrix(matL))

    print("Matrix U")
    print(np.matrix(matU))

    print()

def calculateDiagDeterminal(mat):
    determinant = 1
    for i in range(len(mat)):
        determinant *= mat[i][i]
    return determinant

def solveAndMeasurePerformance(N):
    x = [0 for i in range(N)]
    for i in range(N):
        x[i] = i+1
    
    mat = [[0 for i in range(N)] for j in range(N)] 

    arrU = [0 for i in range(N)] 
    arrL1 = [0 for i in range(N)] 
    arrU1 = [0 for i in range(N)] 
    arrU2 = [0 for i in range(N)] 

    initMatrix(N, mat)
    start = time.perf_counter()

    for i in range(N):
       
        # calc U
        if (i-1 < 0):
            arrU[i] = mat[i][i]
        else:
            arrU[i] = mat[i][i] - arrL1[i-1] * arrU1[i-1]
        
        # calc L1
        if (i+1 < N): 
            arrL1[i] = (mat[i+1][i]) / arrU[i]

        # calc U1
        if (i+1 < N):
            if (i-1 >= 0):
                arrU1[i] = mat[i][i+1] - arrL1[i]*arrU2[i-1]
            else:
                arrU1[i] = mat[i][i+1]

        # calc U2
        if (i+2 < N):
            arrU2[i] = mat[i][i+2]


        #debug(N, arrU, arrU1, arrU2, arrL1)

    end = time.perf_counter()
    delta = (end - start) * 1000

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

    # create original matrix by multiplying L and U
    LU = np.matmul(np.matrix(matL), np.matrix(matU))

    # check for equality
    print("Is original Matrix equal to LU?")
    print(np.allclose(mat, LU, atol=0.1))

    scipyP, scipyL, scipyU = scipy.linalg.lu(mat)
    scipyLU = np.matmul(np.matrix(scipyL), np.matrix(scipyU))

    print("scipy U matrix: ")
    print(np.matrix(scipyU))
    print("my U matrix: ")
    print(np.matrix(matU))

    # calculate determinant
    print("Determinant of original matrix calculated using numpy:")
    print(np.linalg.det(np.matrix(mat)))
    print("Determinant of LU matrix created by multiplying matL and matU calculated using numpy:")
    print(np.linalg.det(LU))
    print("Determinant of LU matrix created using scipy and calculated using numpy:")
    print(1 * np.linalg.det(scipyLU))
    print("Determinant of LU matrix calculated using my function:")
    print(1 * calculateDiagDeterminal(matU))

    return delta

# todo: sprawdz jak sie liczylo wyzacznik L, U skoro nie sa diagonalne
# todo: solve equation x = (1, 2, ..., 124)^T etc

# todo: fix LU solver
solveAndMeasurePerformance(5)

"""
results = []
start = 100
end = 4500
step = 100
for i in range(start, end, step):
    delta = solveAndMeasurePerformance(i)
    results.append(delta)
    print("N: " + str(i) + " Time in ms: " + str(delta))

yPoints = [start - step + i * step for i in range(1, int(end / step))]

plt.plot(yPoints, results, marker='o', linestyle='-')
plt.xlabel('Parametr N')
plt.ylabel('Czas działania funkcji (ms)')
plt.title('Wykres zależności czasu wykonywania od parametru N')
plt.show()
"""






