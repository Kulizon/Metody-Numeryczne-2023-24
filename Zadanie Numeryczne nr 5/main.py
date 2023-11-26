import numpy as np
import matplotlib.pyplot as plt
import math
import time

def vecSubVec(vec1, vec2):
    copy = vec1.copy()
    for i in range(len(vec1)):
        copy[i] = copy[i] - vec2[i]
    return copy

def vectorNorm(a):
    sum = 0
    for i in range(len(a)):
        sum += a[i]*a[i]
    return math.sqrt(sum)

def findApprox(startVector, method, e):
    n = len(startVector)
    
    x = startVector.copy()
    results = [x.copy()]

    if (method != "Gauss" and method != "Jacoby"):
        return -1

    while True:
        xNew = [i + 1 for i in range(len(x))]

        for i in range(n):
            if (i + 1 < n):
                xNew[i] -= x[i+1]
            if (i + 2 < n):
                xNew[i] -= 0.15 * x[i+2]

            if (method == "Jacoby"):
                if (i - 2 >= 0):
                    xNew[i] -= 0.15 * x[i-2]
                if (i - 1 >= 0):
                    xNew[i] -= x[i-1]

            if (method == "Gauss"):
                if (i - 2 >= 0):
                    xNew[i] -= 0.15 * xNew[i-2]
                if (i - 1 >= 0):
                    xNew[i] -= xNew[i-1]
                
            xNew[i] /= 3

        if (vectorNorm(vecSubVec(x, xNew)) < e):
            results.append(xNew.copy())
            break

        x = xNew.copy()
        results.append(xNew.copy())

    return results 

def solve(N, startVector, perfCounting = False, e = 0.000001):
    start = time.perf_counter() * 1000

    jacobyRes = findApprox(startVector, "Jacoby", e)
    gaussRes = findApprox(startVector, "Gauss", e)

    runtime = time.perf_counter() * 1000 - start

    if (not(perfCounting)):
        A = [[0 for _ in range(N)] for _ in range(N)]
        b = [i + 1 for i in range(N)]
        
        for i in range(N):
            for j in range(N):
                if (j == i):
                    A[i][j] = 3
                if (j == i+1 or j == i-1):
                    A[i][j] = 1
                if (j == i+2 or j == i-2):
                    A[i][j] = 0.15
        npRes = np.linalg.solve(A, b)

        xPointsJacoby = [n + 1 for n in range(len(jacobyRes))]
        xPointsGauss = [n + 1 for n in range(len(gaussRes))]

        yPointsJacoby = [vectorNorm(vecSubVec(jacobyRes[i], npRes)) for i in range(len(jacobyRes))]
        yPointsGauss = [vectorNorm(vecSubVec(gaussRes[i], npRes)) for i in range(len(gaussRes))]
        
        np.set_printoptions(precision=5)

        print()
        print("Are Jacoby approximations equal to numpy results?")
        print(np.allclose(jacobyRes[-1], npRes, e))
        #print(list(np.mat(jacobyRes[-1])))
        print()
        print("Are Gauss approximations equal to numpy results?")
        print(np.allclose(gaussRes[-1], npRes, e))
        #print(list(np.mat(gaussRes[-1])))
        print()
    
        return runtime, xPointsJacoby, xPointsGauss, yPointsJacoby, yPointsGauss

    return runtime

val = 0
runtime, xPointsJacoby, xPointsGauss, yPointsJacoby, yPointsGauss = solve(124, [val for i in range(124)], False, 0.000001)


plt.plot(xPointsJacoby, yPointsJacoby, label="metoda Jacobiego")
plt.plot(xPointsGauss, yPointsGauss, label="metoda Gaussa")

plt.yscale("log")
plt.legend()
plt.xlabel('Numer iteracji')
plt.ylabel('Norma różnicy przybliżenia i dokładnego rozwiązania')
plt.title('Wykres różnicy rozwiązania i przybliżenia od numeru iteracji \n dla wektora startowego = ('+ str(val) + ', ' + str(val) + ', ..., ' + str(val) + ')')
plt.show()

# sum = 0
# numOfTests = 10
# for i in range(numOfTests):
#     runtime = solve(124, [1 for _ in range(124)], True)
#     sum += runtime
# print("Średni czas wykonania programu dla N = 124: ")
# print(sum / numOfTests)    
# print()

# results = []
# start = 10
# end = 1000
# step = 10

# for i in range(start, end, step):
#     runtime = solve(i, [1 for _ in range(i)], True)

#     results.append(runtime)
#     print("Czas wykonania dla N=" + str(i) + ": " + str(runtime))

# yPoints = [start + i * step for i in range(1, int(end / step))]

# plt.plot(yPoints, results, marker='o', linestyle='-')
# plt.xlabel('Parametr N')
# plt.ylabel('Czas działania funkcji (ms)')
# plt.title('Wykres zależności czasu wykonywania programu od parametru N')
# plt.show()







