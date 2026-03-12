# can you adapt the program below so after an interval reduction the parabol continues with the interval reduced. 
# for example nog in the 2-3 interval the original parabola is used but is shoyuld be the reduced parabola plotted too
# en vraag 2 bestaat er een wiskundige functie om hier berekeningen mee te doen ? bv hoe stellen we een vgl op die 
# als oplossing wil: voor welke x waarden ligt onze gereduceerde parabool onder bv y=x ?

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
    (3.0, 4.0, 50),   # daarna 10% af in [2,3]
    (5.0, 6.0, 51),    # voorbeeld: nog 5% af
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
