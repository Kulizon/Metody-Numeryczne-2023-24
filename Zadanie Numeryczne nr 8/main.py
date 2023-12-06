
import random
import numpy as np
import matplotlib.pyplot as plt
import math

import pointsData

def approximator(x,functions, a):
    sum = 0
    for i in range(len(functions)):
        sum += a[i] * functions[i](x)
    return sum

def getPointsAsArrays(points):
    xPoints = []
    yPoints = []
    
    if (not(points)):
        return [], []

    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        xPoints.append(x)
        yPoints.append(y)

    return xPoints, yPoints

def solveForCoefficients(points, functions):
    n = len(points)
    m = len(functions)
    A = [[ 0 for _ in range(m)] for _ in range(n)]

    xPoints, yPoints = getPointsAsArrays(points)
    for i in range(n):
        for j in range(m):
            A[i][j] = functions[j](xPoints[i])

    A = np.array(A)
    U, D, V = np.linalg.svd(A)

    U = U[:,:m]
    V = V[:n,:]
    D = np.diag(D)

    Dp = [[0 for _ in range(m)] for _ in range(m)]
    for i in range(len(D)):
        Dp[i][i] = 1/D[i][i]
    Ap = np.transpose(V) @ Dp @ np.transpose(U)
    a = Ap @ yPoints

    return a

def createPlot(points, functions, a, label, exactPoints = None):
    xPoints, yPoints = getPointsAsArrays(points)
    xExactPoints, yExactPoints = getPointsAsArrays(exactPoints)

    yApproximatedPoints = []
    for i in range(len(points)):
        yApproximatedPoints.append(approximator(points[i][0], functions, a))

    plt.plot(xPoints, yPoints, linestyle='', marker='.', label="Dane punkty")
    plt.plot(xPoints, yApproximatedPoints, label="Funkcja przybliżona")

    if (exactPoints):
        plt.plot(xExactPoints, yExactPoints, label="Dokładna funkcja")

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(label)
    plt.show()



###############################################################

# def f1(x):
#     return x**2
# def f2(x):
#     return math.sin(x)
# def f3(x):
#     return math.cos(5*x)
# def f4(x):
#     return math.e**(-x)
# functionsA = [f1, f2, f3, f4]
# pointsA = pointsData.points
# a = solveForCoefficients(pointsA, functionsA)
# createPlot(pointsA, functionsA, a, 'Wykres porównujący funkcję przybliżoną \ndo rozkładu punktów danych w (a)')


###############################################################

# def exactG(x):
#     return 0.01 * x**4 + (-2 * x**3) + 10 * x**2 + (-1 * x)

# def g1(x):
#     return x**4
# def g2(x):
#     return x**3
# def g3(x):
#     return x**2
# def g4(x):
#     return x

# functionsB = [g1, g2, g3, g4]
# pointsB = []
# exactPointsB = []

# for i in range(1000):
#     x = i / 200
#     y = exactG(x)
#     exactPointsB.append((x, y))

# for i in range(100):
#     x = i / 20
#     y = exactG(x)
#     noise = random.randrange(-40, 40) / 10
#     pointsB.append((x, y + noise))

# b = solveForCoefficients(pointsB, functionsB)
# createPlot(pointsB, functionsB, b, 'Wykres porównujący funkcję przybliżoną do rozkładu \npunktów danych dla nowej funckji i funkcji modelującej G(x)', exactPointsB)


###############################################################

def exactH(x):
    return math.sin(x) + 3 * math.cos(math.e**(2/5 * x)) + 1/2 * x**(-2) + 2 * math.log(x)

def h1(x):
    return math.sin(x)
def h2(x):
    return math.cos(math.e**(2/5 * x))
def h3(x):
    return x**(-2)
def h4(x):
    return math.log(x)

functionsC = [h1, h2, h3, h4]
pointsC = []
exactPointsC = []
for i in range(500):
    x = 1 + i / 50
    y = exactH(x)
    exactPointsC.append((x, y))

for i in range(100):
    x = 1 + i / 10
    y = exactH(x)
    noise = random.randrange(-250, 250) / 10
    pointsC.append((x, y + noise))

c = solveForCoefficients(pointsC, functionsC)
createPlot(pointsC, functionsC, c, 'Wykres porównujący funkcję przybliżoną do rozkładu \npunktów danych dla nowej funckji i funkcji modelującej H(x)', exactPointsC)










