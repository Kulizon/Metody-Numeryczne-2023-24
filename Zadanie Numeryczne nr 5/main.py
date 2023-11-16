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

def findApprox(b, method, e = 0.000000000001):
    n = len(b)
    
    prevRes = [b[i] for i in range(n)]
    curRes = [b[i] + 1 for i in range(n)]
    results = [prevRes, curRes]

    if (method != "Gauss" and method != "Jacoby"):
        return -1

    while (vectorNorm(vecSubVec(curRes, prevRes)) > e):
        h = curRes.copy()

        for i in range(n):
            sum = 0
            if (method == "Jacoby"):
                if (i - 1 >= 0):
                    sum += 1 * prevRes[i-1]
                if (i - 2 >= 0):
                    sum += 0.15 * prevRes[i-2]
            if (method == "Gauss"):
                if (i - 1 >= 0):
                    sum += 1 * curRes[i-1]
                if (i - 2 >= 0):
                    sum += 0.15 * curRes[i-2]

            if (i + 1 < n):
                sum += 1 * prevRes[i+1]
            if (i + 2 < n):
                sum += 0.15 * prevRes[i+2]

            curRes[i] = (b[i] - sum) / 3
        prevRes = h
        results.append(curRes)

    return results  

def solve(N, debug):
    start = time.perf_counter() * 1000
    b = [i + 1 for i in range(N)]

    jacobyRes = findApprox(b, "Jacoby")
    gaussRes = findApprox(b, "Gauss")

    runtime = time.perf_counter() * 1000 - start

    A = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if (j == i):
                A[i][j] = 3
            if (j == i+1 or j == i-1):
                A[i][j] = 1
            if (j == i+2 or j == i-2):
                A[i][j] = 0.15
    npRes = np.linalg.solve(A, b)
    print(list(npRes))
    print()

    xPointsJacoby = [n + 1 for n in range(len(jacobyRes))]
    xPointsGauss = [n + 1 for n in range(len(gaussRes))]

    yPointsJacoby = [vectorNorm(vecSubVec(jacobyRes[i], npRes)) for i in range(len(jacobyRes))]
    yPointsGauss = [vectorNorm(vecSubVec(gaussRes[i], npRes)) for i in range(len(gaussRes))]

    if (debug):
        print()
        print("Are Jacoby approximations equal to numpy results?")
        print(np.allclose(jacobyRes[-1], npRes, 0.00001))
        print()
        print("Are Gauss approximations equal to numpy results?")
        print(np.allclose(gaussRes[-1], npRes, 0.00001))
        print()
    
    return runtime, xPointsJacoby, xPointsGauss, yPointsJacoby, yPointsGauss

runtime, xPointsJacoby, xPointsGauss, yPointsJacoby, yPointsGauss = solve(124, True)

plt.plot(xPointsJacoby, yPointsJacoby, label="metoda Jacobiego")
plt.plot(xPointsGauss, yPointsGauss, label="metoda Gaussa")

plt.yscale('log')

plt.legend()
plt.xlabel('Numer iteracji')
plt.ylabel('Norma różnicy przybliżenia i dokładnego rozwiązania')
plt.title('Wykres różnicy rozwiązania i przybliżenia od numeru iteracji')
plt.show()


# todo: wykonaj w kilku punktach startowych, czyli zaczynajac od roznych x



# results = []
# start = 10
# end = 1000
# step = 10

# for i in range(start, end, step):
#     runtime = solve(i, False)

#     results.append(runtime)
#     print("Czas wykonania dla N=" + str(i) + ": " + str(runtime))

# yPoints = [start + i * step for i in range(1, int(end / step))]

# plt.plot(yPoints, results, marker='o', linestyle='-')
# plt.xlabel('Parametr N')
# plt.ylabel('Czas działania funkcji (ms)')
# plt.title('Wykres zależności czasu wykonywania programu od parametru N')
# plt.show()




