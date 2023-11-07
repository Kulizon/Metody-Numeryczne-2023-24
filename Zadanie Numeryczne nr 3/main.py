import numpy as np
import matplotlib.pyplot as plt
import time

def initMatrix(N):
    arrDiag = [0 for i in range(N)] 
    arrUnderDiag1 = [0 for i in range(N)] 
    arrOverDiag1 = [0 for i in range(N)] 
    arrOverDiag2 = [0 for i in range(N)] 

    for i in range(N):
        arrDiag[i] = 1.2
        
        arrUnderDiag1[i] = 0.2 # set 0.2 under diag
        if (i + 1 <= N - 1):
            arrOverDiag1[i] = (0.1)/(i+1) # set 0.1/(i+1) under diag
        if (i + 1 <= N - 2):
            arrOverDiag2[i] = (0.15)/(i+1)**2 # set 0.15/(i+1)**2 under diag

    return arrDiag, arrUnderDiag1, arrOverDiag1, arrOverDiag2

def initDebugMatrix(N):
    mat = [[0 for i in range(N)] for j in range(N)] 
    for i in range(N):
        mat[i][i] = 1.2
        if (i+1 < N):
            mat[i+1][i] = 0.2 # set 0.2 under diag
            mat[i][i+1] = (0.1)/(i+1) # set 0.1/(i+1) under diag
        if (i+2 < N):
            mat[i][i+2] = (0.15)/(i+1)**2 # set 0.15/(i+1)**2 under diag
    
    return mat

def calculateDiagDeterminal(mat):
    determinant = 1
    for i in range(len(mat)):
        determinant *= mat[i][i]
    return determinant

def decomposeMatrixAndMeasurePerformance(arrU, arrL1, arrU1, arrU2, N):
        start = time.perf_counter()

        for i in range(N):
            # calc U
            
            if (i > 0):
                arrU[i] = arrU[i] - arrL1[i-1] * arrU1[i-1]
            
            # calc L1
            arrL1[i] = (arrL1[i]) / arrU[i]

            # calc U1
            if (i > 0):
                arrU1[i] = arrU1[i] - arrL1[i]*arrU2[i-1]

            # calc U2    
            arrU2[i] = arrU2[i]

        end = time.perf_counter()
        delta = (end - start) * 1000

        return delta, arrU, arrL1, arrU1, arrU2

def backsubsitution(x, arrU, arrL1, arrU1, arrU2, N):
    # Ay = x
    # LUy = x ---> Lz = x, Uy = z

    start = time.perf_counter()

    # Lz = x
    z = [0 for i in range(N)]
    for i in range(N):
        tmp = x[i]
        if (i-1 >= 0):
            tmp -= z[i-1] * arrL1[i-1] # this is okay because only non-zero values are at L[i+1][i]
            
        z[i] = tmp

    # Uy = z
    y = np.zeros(N)
    for i in range(N-1, -1, -1):
        tmp = z[i]

        if (i+1 < N):
            tmp -= y[i+1] * arrU1[i] # this is okay because only non-zero values are at U[i][i+1] and at U[i][i+2]
        if (i+2 < N):
            tmp -= y[i+2] * arrU2[i]
            
        y[i] = tmp / arrU[i]

    end = time.perf_counter()
    delta = (end - start) * 1000

    return delta, y


def solve(N, debug=False):
    x = [0 for i in range(N)]
    for i in range(N):
        x[i] = i+1

    arrDiag, arrUnderDiag1, arrOverDiag1, arrOverDiag2 = initMatrix(N)
    
    deltaLU, arrU, arrL1, arrU1, arrU2 = decomposeMatrixAndMeasurePerformance(arrDiag, arrUnderDiag1, arrOverDiag1, arrOverDiag2, N)
    deltaSolve, y = backsubsitution(x, arrU, arrL1, arrU1, arrU2, N)

    if (debug): # only for displaying the results, skip when measuring performance
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

        # calculate determinant
        print()
        print("Determinant of original matrix calculated using numpy:")
        print(np.linalg.det(np.matrix(initDebugMatrix(N))))
        print("Determinant of LU matrix calculated using my function:")
        print(1 * calculateDiagDeterminal(U))
        print()

        # create original matrix by multiplying L and U
        LU = np.matmul(np.matrix(L), np.matrix(U))

        # check for equality
        print("Is original Matrix equal to created LU?")
        print(np.allclose(initDebugMatrix(N), LU, atol=0.001))

        numpyY = np.linalg.solve(LU, x)
        # print("Result Ay = x using back substitution: ")
        # print(y)
        # print("Result Ay = x using numpy:")
        # print(numpyY)
        print("Are my results and numpy results equal?")
        print(np.allclose(y, numpyY, atol=0.001))
        print()

        # print("Original matrix")
        # print(np.matrix(initDebugMatrix(N)))
        # print("LU matrix: ")
        # print(LU)

    return deltaLU + deltaSolve

solve(124, True)

sum = 0
numOfTests = 100
for i in range(numOfTests):
    sum += solve(124)

print("Średni czas wykonania programu dla N = 124: ")
print(sum / numOfTests)    
print()

results = []
start = 10
end = 1000
step = 10
for N in range(start, end, step):
    delta = solve(N)

    results.append(delta)
    print("N: " + str(N) + " Time in ms: " + str(delta))

yPoints = [start + i * step for i in range(1, int(end / step))]

plt.plot(yPoints, results, marker='o', linestyle='-')
plt.xlabel('Parametr N')
plt.ylabel('Czas działania funkcji (ms)')
plt.title('Wykres zależności czasu wykonywania programu od parametru N')
plt.show()






