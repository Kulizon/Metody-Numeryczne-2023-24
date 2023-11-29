import numpy as np
import matplotlib.pyplot as plt
import math

def y(x):
    return 1/(50 + x**2)

def aPoint(n, i):
    return -1 +  2*i/(n+1)

def bPoint(n, i):
    return math.cos((2 * i + 1) / (2 * (n + 1)) * math.pi)

def createPoints(n, createFunction):
    points = []
    for i in range(n):
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

    for j in range(0, n):
        print(cur, [x[j]])
        if (i != j):
            cur = mulPolymonial(cur, [1, -x[j]])
    print()
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

xP = [-1, 0, 1, 2]
yP = [2, 1, 4, 5]

result = p(xP, yP)

print(result)






