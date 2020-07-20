#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

peoples = ['Farrah', 'Fred', 'Felicia']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

fig, axs = plt.subplots()
fig.suptitle("Number of Fruit per Person")

axs.set_ylabel("Quantity of Fruit")
axs.set_ylim(0, 80)
axs.set_yticks(range(0, 81, 10))

for row, qty in enumerate(fruit):
    axs.bar(peoples, qty, width=0.5,
            bottom=[sum(fruit[:row, col]) for col in range(len(qty))],
            label=fruits[row],
            color=colors[row])

axs.legend()
plt.show()
