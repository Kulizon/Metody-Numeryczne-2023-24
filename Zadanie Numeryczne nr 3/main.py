import numpy as np
import matplotlib.pyplot as plt
import time

def initMatrix(N, mat):
    for i in range(N):
        mat[i][i] = 1.2
        if (i+1 < N):
            mat[i+1][i] = 0.2 # set 0.2 under diag
            mat[i][i+1] = (0.1)/(i+1) # set 0.1/(i+1) under diag
        if (i+2 < N):
            mat[i][i+2] = (0.15)/(i+1)**2 # set 0.15/(i+1)**2 under diag

def calculateDiagDeterminal(mat):
    determinant = 1
    for i in range(len(mat)):
        determinant *= mat[i][i]
    return determinant

def decomposeMatrixAndMeasurePerformance(mat, N):
    arrU = [0 for i in range(N)] 
    arrL1 = [0 for i in range(N)] 
    arrU1 = [0 for i in range(N)] 
    arrU2 = [0 for i in range(N)] 

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

    end = time.perf_counter()
    delta = (end - start) * 1000

    return arrU, arrL1, arrU1, arrU2, delta

def backsubsitution(x, matL, matU, N):
    # Ay = x
    # LUy = x ---> Lz = x, Uy = z

    start = time.perf_counter()

    # Lz = x
    L = matL.copy() 
    U = matU.copy() 

    z = [0 for i in range(N)]
    for i in range(N):
        tmp = x[i]
        if (i-1 >= 0):
            tmp -= z[i-1] * L[i][i-1] # this is okay because only non-zero values are at L[i+1][i]
            
        z[i] = tmp / L[i][i]

    # Uy = z
    y = np.zeros(N)
    for i in range(N-1, -1, -1):
        tmp = z[i]

        if (i+1 < N):
            tmp -= y[i+1] * U[i][i+1] # this is okay because only non-zero values are at U[i][i+1] and at U[i][i+2]
        if (i+2 < N):
            tmp -= y[i+2] * U[i][i+2] 
            
        y[i] = tmp / U[i][i]

    end = time.perf_counter()
    delta = (end - start) * 1000

    return y, delta


def solve(N):
    x = [0 for i in range(N)]
    for i in range(N):
        x[i] = i+1

    mat = [[0 for i in range(N)] for j in range(N)] 
    initMatrix(N, mat)
    
    arrU, arrL1, arrU1, arrU2, deltaLU = decomposeMatrixAndMeasurePerformance(mat, N)

    # create L matrix
    L = [[0 for i in range(N)] for j in range(N)] 
    for i in range(N):
        L[i][i] = 1
        if (i+1 < N):
            L[i+1][i] = arrL1[i]

    # create U matrix
    U = [[0 for i in range(N)] for j in range(N)] 
    for i in range(N):
        U[i][i] = arrU[i]
        if (i+1 < N):
            U[i][i+1] = arrU1[i]
        if (i+2 < N):
            U[i][i+2] = arrU2[i]

    y, deltaSolve = backsubsitution(x, L, U, N)

    # create original matrix by multiplying L and U
    LU = np.matmul(np.matrix(L), np.matrix(U))

    # check for equality
    print("Is original Matrix equal to created LU?")
    print(np.allclose(mat, LU, atol=0.01))

    # print("Original matrix")
    # print(np.matrix(mat))
    # print("LU matrix: ")
    # print(LU)

    # calculate determinant
    print()
    print("Determinant of original matrix calculated using numpy:")
    print(np.linalg.det(np.matrix(mat)))
    print("Determinant of LU matrix calculated using my function:")
    print(1 * calculateDiagDeterminal(U))
    print()

    
    numpyY = np.linalg.solve(LU, x)
    print("Result Ay = x using back substitution: ")
    print(y)
    print("Result Ay = x using numpy:")
    print(numpyY)
    print("Are my results and numpy results equal?")
    print(np.allclose(y, numpyY, atol=0.01))
    print()

    return deltaLU + deltaSolve

solve(4)

"""
results = []
start = 10
end = 1500
step = 10
for N in range(start, end, step):
    mat = [[0 for i in range(N)] for j in range(N)] 
    initMatrix(N, mat)
    L, U, delta = decomposeMatrixAndMeasurePerformance(mat, N)

    results.append(delta)
    print("N: " + str(N) + " Time in ms: " + str(delta))

yPoints = [start + i * step for i in range(1, int(end / step))]

plt.plot(yPoints, results, marker='o', linestyle='-')
plt.xlabel('Parametr N')
plt.ylabel('Czas działania funkcji (ms)')
plt.title('Wykres zależności czasu wykonywania od parametru N')
plt.show()
"""






