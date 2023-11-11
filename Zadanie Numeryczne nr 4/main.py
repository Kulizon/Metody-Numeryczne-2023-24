import time
import numpy as np
import matplotlib.pyplot as plt

def backsubsitution(b, u, arrDiag, arrBandOverDiag1, N):
    # Az = b
    z = [0 for i in range(N)]
    for i in range(N-1, -1, -1):
        tmp = b[i]

        if (i+1 < N):
            tmp -= z[i+1] * arrBandOverDiag1[i] 
            
        z[i] = tmp / arrDiag[i]

    y = [0 for i in range(N)]
    for i in range(N-1, -1, -1):
        tmp = u[i]

        if (i+1 < N):
            tmp -= y[i+1] * arrBandOverDiag1[i] 
            
        y[i] = tmp / arrDiag[i]



    return z, y

def sumVec(vec, N):
    sum = 0
    for i in range(N):
        sum += vec[i]
    return sum

def scalarMulVec(scalar, vec, N):
    copy = [vec[i] for i in range(N)]
    for i in range(N):
        copy[i] = scalar * vec[i]
    return copy

def vecSubVec(vec1, vec2, N):
    copy = [vec1[i] for i in range(N)]
    for i in range(N):
        copy[i] = copy[i] - vec2[i]
    return copy


def solve(N, debug):
    start = time.perf_counter()

    b = [5 for _ in range(N)]
    arrDiag = [11 for _ in range(N)]
    arrBandOverDiag1 = [7 for _ in range(N)]

    u = [1 for _ in range(N)]
    #vt = [1 for _ in range(N)] -> nieistotne, bo mnozac przez to jedyne co robimy to dodajemy wszystkie elementy wektora (bo elementy v^t to jedynki)

    # backsubstitution for Az = b and Ay = u
    z, y = backsubsitution(b, u, arrDiag, arrBandOverDiag1, N)

    # x = z - [(v^t * z) * y] / [1 + v^t * y]  
    #          [liczba * wektor] / [liczba]
    
    alfa = sumVec(z, N) # v^t * z
    beta = sumVec(y, N) # v^t * y

    gamma = (alfa)/(1 + beta)

    result = scalarMulVec(gamma, y, N)
    x = vecSubVec(z, result, N)

    end = time.perf_counter()
    delta = (end - start) * 1000

    if (debug):
        mat = [[1 for j in range(N)] for i in range(N)]

        for i in range(N):
            mat[i][i] = 12
            if (i + 1 < N):
                mat[i][i+1] = 8
        
        numpyResult = np.linalg.solve(mat, b)

        print("Is my result equal to numpy result?")
        # print(numpyResult)
        # print(x)
        print(np.allclose(numpyResult, x, 0.00001))

    return delta

solve(80, True)

sum = 0
numOfTests = 100
for i in range(numOfTests):
    sum += solve(80, False)
print("Średni czas wykonania programu dla N = 80: ")
print(sum / numOfTests)    
print()

results = []
start = 100
end = 10000
step = 100

for i in range(start, end, step):
    runtime = solve(i, False)

    results.append(runtime)
    print("Czas wykonania dla N=" + str(i) + ": " + str(runtime))

yPoints = [start + i * step for i in range(1, int(end / step))]

plt.plot(yPoints, results, marker='o', linestyle='-')
plt.xlabel('Parametr N')
plt.ylabel('Czas działania funkcji (ms)')
plt.title('Wykres zależności czasu wykonywania programu od parametru N')
plt.show()


