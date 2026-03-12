
import numpy as np
import matplotlib.pyplot as plt

# --- Instellingen ---------------------------------------------------------
# Vul hier de parabool-formule in: y = a*x^2 + b*x + c
a, b, c = 1.0, 0.0, 0.0          # standaard: y = x^2

# De x-waardes die je wil doorlopen
x_values = np.linspace(0, 1, 10)

# Percentages die je van Y wil aftrekken op opeenvolgende punten.
# 10 betekent: y wordt y * (1 - 0.10)
modifiers = [10, 20, 5, 40, 3, 15]   
# Worden herhaald indien lijst te kort is
# --------------------------------------------------------------------------


# De originele parabool
def parabola(x):
    return a*x**2 + b*x + c

y_original = parabola(x_values)
y_modified = y_original.copy()

# Pas de y-waarde aan volgens de percentages
for i, pct in enumerate(modifiers):
    if i < len(y_modified):
        y_modified[i] = y_modified[i] * (1 - pct/100.0)

# Plotten
plt.figure(figsize=(10,6))
plt.plot(x_values, y_original, label="Originele parabool")
plt.scatter(x_values[:len(modifiers)], y_modified[:len(modifiers)],
            label="Aangepaste punten", zorder=5)

plt.plot(x_values, y_modified, '--', label="Parabool verdergetrokken")
plt.title("Parabool met aangepaste Y-punten")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parabool: definieer zelf ----------------
a, b, c = 1.0, 0.0, 0.0      # y = x^2 voorbeeld

def parabola(x):
    return a*x**2 + b*x + c

# ---------------- Intervallen + reducties ----------------
#   (x_start, x_end, percentage)
intervals = [
    (1.0, 2.0, 20),   # 20% af in [1,2]
    (2.0, 3.0, 10),   # daarna 10% af in [2,3]
    (3.0, 4.0, 5),    # voorbeeld: nog 5% af
]

# ---------------- Domein ----------------
x = np.linspace(0, 5, 500)
y = parabola(x)
y_modified = y.copy()

# ---------------- Verwerking per interval ----------------
for (xs, xe, pct) in intervals:
    mask = (x >= xs) & (x < xe)
    y_modified[mask] *= (1 - pct/100.0)

# ---------------- Plot ----------------
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Originele parabool")
plt.plot(x, y_modified, label="Aangepaste parabool", linestyle="--")
plt.axvline(1, color='grey', alpha=0.4)
plt.axvline(2, color='grey', alpha=0.4)
plt.axvline(3, color='grey', alpha=0.4)
plt.axvline(4, color='grey', alpha=0.4)

plt.grid(True)
plt.legend()
plt.title("Parabool met interval-gebaseerde cumulatieve reducties")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
