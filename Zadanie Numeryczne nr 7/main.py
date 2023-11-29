import numpy as np
import matplotlib.pyplot as plt
import math

def y(x):
    return 1/(1 + 50 * x**2)

def poly(x, a):
    val = 0
    n = len(a) - 1
    for i in range(len(a)):
        val += a[i] * x**n
        n -= 1
    return val


def aPoint(n, i):
    return -1 + 2 * (i/(n+1))

def bPoint(n, i):
    return math.cos((2 * i + 1) / (2 * (n + 1)) * math.pi)

def createPointsX(n, createFunction):
    points = []
    for i in range(n + 1):
        points.append(createFunction(n, i))
    return points

# p(x) - wielomian interpolacyjny
# p(x) = suma od i = 0 do n (y_i * l_i(x))
# l_i(x) = iloczyn od j = 0 do n oraz j != i ((x-x_j) / (x_i - x_j))

def mulPolymonial(poly1, poly2):
    degree_poly1 = len(poly1)
    degree_poly2 = len(poly2)
    result = [0] * (degree_poly1 + degree_poly2 - 1)

    for i in range(degree_poly1):
        for j in range(degree_poly2):
            result[i + j] += poly1[i] * poly2[j]

    return result

def l(x, i):
    n = len(x)
    divider = 1
    for j in range(n):
        if (i != j):
            divider *= x[i] - x[j]

    cur = [1]

    for j in range(n):
        if (i != j):
            cur = mulPolymonial(cur, [1, -x[j]])
    return np.array(cur) * 1/divider
    
def p(xP, yP):
    n = len(xP)
    L = []
    for i in range(n):
        L.append(l(xP, i))

    for i in range(n):
        L[i] = np.array(L[i]) * yP[i]

    sum = np.zeros(n)
    for i in range(n):
        sum = sum + L[i]

    return sum # nasz wielomian interpolacyjny

def extractPolymonalEquation(polymonal):
    text = ""
    n = len(polymonal) - 1
    for i in range(len(polymonal)):
        if polymonal[i] == 0:
            continue
        text += str(round(polymonal[i], 2)) + " * x^{" + str(n) + "} + "
        n -= 1
    text += "0"
    return text


# for i in range(N):
#     print("Dla xP = " + str(xP[i]))
#     print("Dla yP = " + str(y(xP[i])))
#     print("Dla yP interpolowane = " + str(poly(xP[i], inter)))
#     print()   

def createComparePlot(interpolationPoly, method):
    xFinal = [-1 + i / 100 for i in range(200)]
    yFinal = [y(xi) for xi in xFinal]
    yFinalInter = [poly(xi, interpolationPoly) for xi in xFinal]

    plt.plot(xFinal, yFinal, label="Funkcja 1/(1 + 50x^2)")
    plt.plot(xFinal, yFinalInter, label="Interpolacja")

    plt.xlim(-1, 1)  
    plt.ylim(-2, 2)  

    plt.legend()
    plt.xlabel('Oś x')
    plt.ylabel('Oś y')
    plt.title("Porównanie funkcji 1/(1 + 50x^2) z funkcją interpolowną \nwzorami Lagrange'a dla N = " + str(len(interpolationPoly)-1) + " węzłów interpolacyjnych i metody (" + method + ")")
    plt.show()


def createInterpolation(N, method):
    if (method != "a" and method != "b"):
        print("Zła metoda!")
        return

    xP = createPointsX(N, bPoint if method == "b" else aPoint)
    yP = [y(xP[i]) for i in range(N + 1)]
    interpolationPoly = p(xP, yP)
    createComparePlot(interpolationPoly, method)
    #print(extractPolymonalEquation(inter))


createInterpolation(30, "b")









