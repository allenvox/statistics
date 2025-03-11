import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

P1 = np.array([
    [0.7, 0.06, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    [0.03, 0.7, 0.06, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    [0.03, 0.03, 0.7, 0.06, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    [0.03, 0.03, 0.03, 0.7, 0.06, 0.03, 0.03, 0.03, 0.03, 0.03],
    [0.03, 0.03, 0.03, 0.03, 0.7, 0.06, 0.03, 0.03, 0.03, 0.03],
    [0.03, 0.03, 0.03, 0.03, 0.03, 0.7, 0.06, 0.03, 0.03, 0.03],
    [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.7, 0.06, 0.03, 0.03],
    [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.7, 0.06, 0.03],
    [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.7, 0.06],
    [0.06, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.7]
])

P2 = np.full((10, 10), 0.1)

print("Проверка P1:")
print("Сумма по строкам:", [round(s, 2) for s in np.sum(P1, axis=1).tolist()])
print("Сумма по столбцам:", [round(s, 2) for s in np.sum(P1, axis=0).tolist()])
print("Проверка P2:")
print("Сумма по строкам:", [round(s, 2) for s in np.sum(P2, axis=1).tolist()])
print("Сумма по столбцам:", [round(s, 2) for s in np.sum(P2, axis=0).tolist()])

def generate_markov_chain(P, n_steps, initial_state=0):
    states = [initial_state]
    values = []
    n_states = P.shape[0]
    
    for _ in range(n_steps):
        val = np.random.exponential(scale=1.0)
        values.append(val)
        current_state = states[-1]
        next_state = np.random.choice(range(n_states), p=P[current_state])
        states.append(next_state)
    
    values = np.array(values)
    values = (values - values.min()) / (values.max() - values.min())
    return states, values

n_steps = 100
states1, values1 = generate_markov_chain(P1, n_steps)
states2, values2 = generate_markov_chain(P2, n_steps)

freq1 = np.bincount(states1, minlength=10) / len(states1)
freq2 = np.bincount(states2, minlength=10) / len(states2)

print("\nМатрица P1 (10×10):")
print(P1)
print("Сгенерированные значения:", values1.tolist())
print("Переходы:", [int(s) for s in states1])
print("Частоты состояний:", freq1.tolist())

print("\nМатрица P2 (10×10):")
print(P2)
print("Сгенерированные значения:", values2.tolist())
print("Переходы:", [int(s) for s in states2])
print("Частоты состояний:", freq2.tolist())

plt.figure(figsize=(12, 6))
plt.plot(values1, label="Цепь 1 (P1, 10×10)", alpha=0.7)
plt.plot(values2, label="Цепь 2 (P2, 10×10)", alpha=0.7)
plt.title("Сравнение поведения двух цепей Маркова")
plt.xlabel("Шаг")
plt.ylabel("Нормированное значение")
plt.legend()
plt.grid(True)
plt.show()

def autocorrelation(x, max_lag=20):
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    acf = []
    for lag in range(max_lag + 1):
        cov = np.sum((x[:n-lag] - mean) * (x[lag:] - mean)) / n
        acf.append(cov / var)
    return np.array(acf)

acf1 = autocorrelation(values1)
acf2 = autocorrelation(values2)

plt.figure(figsize=(12, 6))
plt.plot(acf1, label="Автокорреляция Цепи 1 (P1, 10×10)", marker='o')
plt.plot(acf2, label="Автокорреляция Цепи 2 (P2, 10×10)", marker='o')
plt.title("Автокорреляция цепей Маркова")
plt.xlabel("Лаг")
plt.ylabel("Значение автокорреляции")
plt.legend()
plt.grid(True)
plt.show()
