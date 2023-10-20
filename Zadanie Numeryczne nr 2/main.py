import random
import numpy as np
import matplotlib as mpl

A1 = np.matrix([[2.554219275, 0.871733993, 0.052575899, 0.240740262, 0.316022841],
              [0.871733993, 0.553460938, -0.070921727, 0.255463951, 0.707334556],
              [0.052575899, -0.070921727, 3.409888776, 0.293510439, 0.847758171],
              [0.240740262, 0.255463951, 0.293510439, 1.108336850, -0.206925123],
              [0.316022841, 0.707334556, 0.847758171, -0.206925123, 2.374094162]])

A2 = np.matrix([[2.645152285, 0.544589368, 0.009976745, 0.327869824, 0.424193304],
              [0.544589368, 1.730410927, 0.082334875, -0.057997220, 0.318175706],
              [0.009976745, 0.082334875, 3.429845092, 0.252693077, 0.797083832],
              [0.327869824, -0.057997220, 0.252693077, 1.191822050, -0.103279098],
              [0.424193304, 0.318175706, 0.797083832, -0.103279098, 2.502769647]])

b = np.matrix([[-0.642912346], [-1.408195475], [4.595622394], [-5.073473196], [2.178020609]])

# między 0.0000005 a 0.0000012
e = ((random.random() * 0.5) + 0.8) * 10**(-5)
bDist = b + e

outcomeA1 = np.linalg.solve(A1, b)
outcomeA1dist = np.linalg.solve(A1, bDist)

outcomeA2 = np.linalg.solve(A2, b)
outcomeA2dist = np.linalg.solve(A2, bDist)


print("Wynik A1*y = b")
print(outcomeA1)
print("\nWynik A1*y = b + Δb")
print(outcomeA1dist)
print("\n======== Błąd |A1/b - A1/(b + Δb) ========")
print(np.abs(outcomeA1 - outcomeA1dist))

print("\n\nWynik A2*y = b ")
print(outcomeA2)
print("\nWynik A2*y = b + Δb")
print(outcomeA2dist)
print("\n======== Błąd |A2/b - A2/(b + Δb) ========")
print(np.abs(outcomeA2 - outcomeA2dist))
print("\n")