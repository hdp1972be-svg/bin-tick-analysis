
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parabool: definieer zelf ----------------
a, b, c = 1.0, 0.0, 0.0      # y = x^2

def parabola(x):
    return a*x**2 + b*x + c

def line(x):
    return x

# ---------------- Intervallen + reducties ----------------
# (x_start, x_end, percentage)
intervals = [
    (2.0, 3.0, 10),
    (4.0, 5.0, 10),
    (6.0, 7.0, 20),
    (8.0, 9.0, 20),
]

# ---------------- Domein ----------------
x = np.linspace(0, 10, 700)
y_original = parabola(x)
y_line = line(x)

# Cumulatieve curve
y_mod = y_original.copy()

# ---------------- Cumulatieve verwerking ----------------
for (xs, xe, pct) in intervals:
    mask = (x >= xs) & (x < xe)
    reduction_factor = (1 - pct/100.0)
    
    # Pas y binnen het interval aan
    y_mod[mask] *= reduction_factor
    
    # --- TRICK: alles NA het interval moet verdergaan
    # met de verlaagde uitgangspositie
    tail_mask = (x >= xe)
    if np.any(tail_mask):
        # hoeveel is de shift op het einde van het interval?
        idx = np.where(x < xe)[0][-1]
        delta = y_mod[idx] - y_original[idx]
        y_mod[tail_mask] = y_original[tail_mask] + delta

# ---------------- Plot ----------------
plt.figure(figsize=(10, 6))
plt.plot(x, y_original, label="Originele parabool")
plt.plot(x, y_line, label="Lineair verlies")
plt.plot(x, y_mod, '--', label="Gecumuleerde gereduceerde parabool", linewidth=2)

# verticale lijnen voor intervallen
for (xs, xe, pct) in intervals:
    plt.axvline(xs, color='grey', alpha=0.3)
    plt.axvline(xe, color='grey', alpha=0.3)

plt.grid(True)
plt.legend()
plt.title("Parabool met cumulatieve intervalreducties")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
