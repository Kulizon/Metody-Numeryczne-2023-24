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

def maxEigenValueNumpy(A):
    eigenValues = np.linalg.eigvals(A)
    maxEigenValue = np.max(np.abs(eigenValues)) 
    return maxEigenValue

def eigenValuePowerMethod(A):
    curVec = [1 for _ in range(len(A))]
    resultVecs = [curVec]
    resultLambdas = [1]

    while(True):
        curVec = matMulVec(A, curVec)
        lamb, normedVec = normalize(curVec)
        curVec = normedVec

        resultVecs.append(normedVec)
        resultLambdas.append(lamb)

        # jednak pewnie bedzie wolno zbiegac wiec nalezy przesunac A - p * 1

        diff = abs(vecNorm(resultVecs[len(resultVecs) - 1]) - vecNorm(resultVecs[len(resultVecs) - 2]))
        if (diff < 0.00000001):
            break

    eigenVec = resultVecs[len(resultVecs) - 1]
    eigenVal = resultLambdas[len(resultLambdas) - 1]

    print(eigenVec)
    print(len(resultLambdas))
    print(eigenVal, maxEigenValueNumpy(A))

def eigenValueQrMethod(A):
    curMat = A.copy()
    resultMats = [curMat]

    underDiagSum = curMat[1][0] + curMat[1][2] + curMat[2][3]
    while(abs(underDiagSum) > 0.00001):
        # wykorzstaj biblioteczny rozkłąd QR
        Q, R = np.linalg.qr(curMat)

        nextMat = np.matmul(R, Q) # todo: zrob w O(1) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        curMat = nextMat.copy()
        resultMats.append(curMat)

        underDiagSum = curMat[1][0] + curMat[1][2] + curMat[2][3]

        # jak bedzie podobna do macierzy trojkatnej gornej to stop, wektory wlasne beda wtedy na diagonali

    resultLambdas = []
    finalMat = resultMats[len(resultMats) - 1]
    for i in range(len(A)):
        resultLambdas.append(finalMat[i][i])
    
    print(resultLambdas)

A = [[8, 1, 0, 0],
     [1, 7, 2, 0],
     [0, 2, 6, 3],
     [0, 0, 3, 5]]
# eigenValuePowerMethod(A)
eigenValueQrMethod(A)

def solve():
    return 1






