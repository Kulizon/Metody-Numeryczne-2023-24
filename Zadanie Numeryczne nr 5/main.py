import numpy as np
import matplotlib.pyplot as plt
import math

def vecSubVec(vec1, vec2):
    copy = [vec1[i] for i in range(len(vec1))]
    for i in range(len(vec1)):
        copy[i] = copy[i] - vec2[i]
    return copy

def vectorNorm(a):
    sum = 0
    for i in range(len(a)):
        sum += a[i]*a[i]
    return math.sqrt(sum)

def Jacoby(b, e = 0.0000001):
    n = len(b)

    prevRes = [1 for _ in range(n)]
    curRes = [2 for _ in range(n)]

    while (vectorNorm(vecSubVec(curRes, prevRes)) > e):
        h = curRes.copy()

        for i in range(n):
            sum = 0
            if (i - 1 >= 0):
                sum += 1 * prevRes[i-1]

            if (i - 2 >= 0):
                sum += 0.15 * prevRes[i-2]

            if (i + 1 < n):
                sum += 1 * prevRes[i+1]

            if (i + 2 < n):
                sum += 0.15 * prevRes[i+2]

            curRes[i] = (b[i] - sum) / 3

        prevRes = h

    return curRes

def Gauss():
    return 1

def solve(N, debug):
    b = [i + 1 for i in range(N)]

    print(Jacoby(b))
    if (debug):
        A = [[0 for _ in range(N)] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if (j == i):
                    A[i][j] = 3
                if (j == i+1 or j == i-1):
                    A[i][j] = 1
                if (j == i+2 or j == i-2):
                    A[i][j] = 0.15

        print(np.linalg.solve(A, b))

solve(5, True)

# todo: wykonaj w kilku punktach startowych, czyli zaczynajac od roznych x


