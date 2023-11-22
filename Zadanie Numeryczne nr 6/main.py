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

def matMulVec(mat, vec): # TODO: zrob to w O(1) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

    finalEigenVec = resultVecs[len(resultVecs) - 1]
    finalEigenVal = resultLambdas[len(resultLambdas) - 1]
    
    numpyEigenValues, numpyEigenVectors = np.linalg.eig(A)
    print("Czy moja maksymalna wartość własna obliczona metodą potęgową zgadza się z numpy?")
    print(abs(max(numpyEigenValues) - finalEigenVal) < 0.000001)
    print()

    finalNumpyMaxEigenValVec = []
    for i in range(4):
        finalNumpyMaxEigenValVec.append(numpyEigenVectors[i][0] / (-0.62856775)) # convert to my base
    print("Czy mój wektor odpowiadający największej wartości własnej zgadza się z numpy?")
    print(np.allclose(finalEigenVec, finalNumpyMaxEigenValVec, 0.000001))
    print()

    return resultVecs, resultLambdas

def getDiagElements(A):
    diag = []
    for i in range(len(A)):
        diag.append(A[i][i])
    return diag

def eigenValueQrMethod(A):
    curMat = A.copy()
    resultMats = [curMat]
    resultLambdas = [getDiagElements(curMat)]

    underDiagSum = curMat[1][0] + curMat[1][2] + curMat[2][3]
    while(abs(underDiagSum) > 0.00001):
        Q, R = np.linalg.qr(curMat)

        nextMat = np.matmul(R, Q) # TODO: zrob w O(1) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        curMat = nextMat.copy()
        resultMats.append(curMat)
        resultLambdas.append(getDiagElements(curMat))

        underDiagSum = curMat[1][0] + curMat[1][2] + curMat[2][3]
        # jak bedzie podobna do macierzy trojkatnej gornej to stop, wektory wlasne beda wtedy na diagonali
    
    finalEigenVals = resultLambdas[len(resultLambdas)-1]

    numpyEigenValues, numpyEigenVectors = np.linalg.eig(A)
    print("Czy moje wartości własne obliczone metodą rozkładu QR zgadzają się z numpy?")
    print(np.allclose(finalEigenVals, numpyEigenValues, 0.000001))
    print()

    return resultMats, resultLambdas

def createPowerMethodPlot(eigenVals, A):
    xPointsPowerMethod = []
    yPointsPowerMethod = []
    for i in range(len(eigenVals)):
        xPointsPowerMethod.append(i+1)
        yPointsPowerMethod.append(abs(eigenVals[i] - maxEigenValueNumpy(A)))

    plt.plot(xPointsPowerMethod, yPointsPowerMethod, label="Metoda Potęgowa")

    plt.yscale("log")
    plt.legend()
    plt.xlabel('Numer iteracji')
    plt.ylabel('Różnicy przybliżenia i dokładnej wartości własnej')
    plt.title('TODO: dodaj tytuł')
    plt.show()

def createQrMethodPlot(eigenVals):

    exactEigenValues =  eigenVals.pop()

    xPoints = []
    yPoints = []

    for i in range(len(exactEigenValues)):
        exactEigenVal = exactEigenValues[i]
        xPoints.append([])
        yPoints.append([])
        for j in range(len(eigenVals)):
            xPoints[i].append(j)
            yPoints[i].append(abs(exactEigenVal - eigenVals[j][i]))

    plt.plot(xPoints[0], yPoints[0], label="Wartość własna 1")
    plt.plot(xPoints[1], yPoints[1], label="Wartość własna 2")
    plt.plot(xPoints[2], yPoints[2], label="Wartość własna 3")
    plt.plot(xPoints[3], yPoints[3], label="Wartość własna 4")

    plt.yscale("log")
    plt.legend()
    plt.xlabel('Numer iteracji')
    plt.ylabel('Różnicy przybliżenia i dokładnej wartości własnej')
    plt.title('TODO: dodaj tytuł')
    plt.show()
    
    # zapisz ostateczne wyniki w jakiejs tablicy
    # potem wykorzystaj te wyniki w tworzeniu wykresow
    # dla kazdej wartosci wlasnej (4) zrob wykres - roznica dokladnego rozwiazania z aktualnym zaleznie od iteracji i

A = [[8, 1, 0, 0],
     [1, 7, 2, 0],
     [0, 2, 6, 3],
     [0, 0, 3, 5]]
powerMethodEigenVecs, powerMethodEigenVals = eigenValuePowerMethod(A)
qrMetodMats, qrMethodEigenVals = eigenValueQrMethod(A)

# createPowerMethodPlot(powerMethodEigenVals, A)
# createQrMethodPlot(qrMethodEigenVals)







