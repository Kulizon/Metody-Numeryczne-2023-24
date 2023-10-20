import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x**2)

def df(x):
    return 2 * x * np.cos(x**2)

def D1(h, x):
    return (f(x + h) - f(x) ) /  h

def D2(h, x):
    return (f(x + h) - f(x - h)) / (2 * h)

def er1(h, x):
    return np.abs(D1(h, x) - df(x))

def er2(h, x):
    return np.abs(D2(h, x) - df(x))


h_values = np.logspace(-16, 0, 600)
x = 0.2

errors1_float = [er1(np.float32(h), np.float32(x)) for h in h_values]
errors1_double = [er1(np.float64(h), np.float64(x)) for h in h_values]
errors2_float = [er2(np.float32(h), np.float32(x)) for h in h_values]
errors2_double = [er2(np.float64(h), np.float64(x)) for h in h_values]

plt.figure(figsize=(10, 7))

plt.loglog(h_values, errors1_float, label="błąd wzoru (a) dla float")
plt.loglog(h_values, errors1_double, label="błąd wzoru (a) dla double")
plt.loglog(h_values, errors2_float, label="błąd wzoru (b) dla float")
plt.loglog(h_values, errors2_double, label="błąd wzoru (b) dla double")

plt.xlabel("wartość h")
plt.ylabel("|Dhf(x) - f'(x)|")
plt.legend()
plt.title("Błąd przybliżenia pochodnej dla f(x) = sin(x^2) i x = 0.2")
plt.grid(True)
plt.show()