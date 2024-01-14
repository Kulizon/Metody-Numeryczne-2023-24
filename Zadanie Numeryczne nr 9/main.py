
import random
import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    return math.sin(x) - 0.4

def g(x):
    return (math.sin(x) - 0.4)**2

def df(x):
    return math.cos(x)

def dg(x):
    return 2*(math.sin(x) - 0.4)*math.cos(x)


x1 = 0
x2 = math.pi/2
e = 0.000000001
labels = ["bisekcji", "falsi", "siecznych", "Newtona"]

def method1(a, b, func):
    c = (a+b)/2
    points = [c]

    while(abs(a - c) > e):
        aVal = func(a)
        bVal = func(b)
        cVal = func(c)

        if aVal * cVal < 0:
            b = c
        elif cVal * bVal < 0:
            a = c

        c = (a+b)/2
        points.append(c)

    return points

def method2(a, b, func):
    c = (a*func(b) - b*func(a))/(func(b) - func(a))
    points = [c]

    while(abs(a - c) > e):
        aVal = func(a)
        bVal = func(b)
        cVal = func(c)

        if aVal * cVal < 0:
            b = c
        elif cVal * bVal < 0:
            a = c

        c = (a*func(b) - b*func(a)) / (func(b) - func(a))
        points.append(c)

    return points

def method3(a, b, func):
    c = (a*func(b) - b*func(a))/(func(b) - func(a))
    points = [c]

    while(abs(a - c) > e):
        b = a
        a = c 

        c = (a*func(b) - b*func(a)) / (func(b) - func(a))
        points.append(c)

    return points

def method4(xi, func, dFunc):
    xii = xi - func(xi)/dFunc(xi)
    points = [xi, xii]

    while(abs(xii-xi) > e):
        xi = xii
        xii = xi - func(xi)/dFunc(xi)
        points.append(xii)
        
    return points

def createPlot(data, func, offset = 0):
    xExact = math.asin(0.4)

    plt.figure(figsize=(7, 6))
    for i in range(len(data)):
        x = data[i]

        y = [abs(func(x[i])) for i in range(len(x))]
        plt.plot([j for j in range(len(data[i]))], y, label="Metoda " + labels[i + offset])

    plt.yscale("log")
    plt.legend()
    plt.xlabel('liczba iteracji i')
    plt.ylabel('błąd log|x* - x_i')
    plt.title("Porównanie metod znajdywania rozwiązywania \nukładu " + func.__name__ + "(x)=0 na przedziale [0, π/2]")
    plt.show()


def iterationPrint(data, offset = 0):
    for i in range(len(data)):
        print("Dla metody " + str(labels[i + offset]) + " potrzeba " + str(len(data[i])) + " iteracji.")

def resultPrint(data, offset = 0):
    for i in range(len(data)):
        print("Dla metody " + str(labels[i + offset]) + " wynik to: " + str(data[i][-1]) + ".")


f1 = method1(x1, x2, f)
f2 = method2(x1, x2, f)
f3 = method3(x1, x2, f)
f4 = method4(1/2, f, df)
dataF = [f1, f2, f3, f4]

print("============================= Funkcja f(x) =============================")
iterationPrint(dataF)
print()
resultPrint(dataF)
createPlot(dataF, f)


###p1 = method1(x1, x2, g) ----> niemożliwe jest wykonanie metody bisekcji bo nie jest spełniony warunek różnych znaków g(a) * g(b) < 0 (bo do kwadratu jest)
###p2 = method2(x1, x2, g) ----> to samo co u góry
g3 = method3(x1, x2, g) # bardzo wolna
g4 = method4(1/2, g, dg) # też bardzo wolna można jakoś usprawnić niby  
dataG = [g3, g4]

print("============================= Funkcja g(x) =============================")
iterationPrint(dataG, 2)
print()
resultPrint(dataG, 2)
createPlot(dataG, g, 2)


# można usprawnić metodę Newtona dla pierwiastków wielokrotnych dzieląc 
# sobie funkcję przez jej pochodną i taka funckja ma tylko pierwiastki jednokrotne

def h(x):
    return g(x)/dg(x) # w ten sposób możemy usprawnić, bo taka funkcja ma na pewno jednokrotne miejsca zerowe

def dh(x):
    return 1/2 + (5 * (math.sin(x))**2 - 2 * math.sin(x))/(10*(math.cos(x))**2)

def compareUpgradedMethodToOriginal(g4, h4):
    xExact = math.asin(0.4)

    y = [abs(g(g4[i])) for i in range(len(g4))]
    plt.plot([j for j in range(len(g4))], y, label="Metoda Newtona dla g(x)=0")

    y = [abs(h(h4[i])) for i in range(len(h4))]
    plt.plot([j for j in range(len(h4))], y, label="Metoda Newtona dla h(x)=0")

    plt.xlim(0, 50)
    plt.yscale("log")
    plt.legend()
    plt.xlabel('liczba iteracji i')
    plt.ylabel('błąd log|x* - x_i')
    plt.title("Porównanie metody Newtona dla układu g(x)=0 i h(x)=0")
    plt.show()


print("============================= Funkcja h(x) = g(x)/g'(x) =============================")
h4 = method4(1/2, h, dh) 
iterationPrint([h4])
print()
resultPrint([h4])
compareUpgradedMethodToOriginal(g4, h4[:-1])