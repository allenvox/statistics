import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

P1 = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8]
])

P2 = np.array([
    [0.2, 0.4, 0.4],
    [0.4, 0.2, 0.4],
    [0.4, 0.4, 0.2]
])

print("Проверка P1:")
print("Сумма по строкам:", np.sum(P1, axis=1).tolist())
print("Сумма по столбцам:", np.sum(P1, axis=0).tolist())
print("Проверка P2:")
print("Сумма по строкам:", np.sum(P2, axis=1).tolist())
print("Сумма по столбцам:", np.sum(P2, axis=0).tolist())

def generate_markov_chain(P, n_steps, initial_state=0):
    states = [initial_state]  # Список состояний
    values = []  # Сгенерированные значения
    
    for _ in range(n_steps):
        # Генерируем случайное значение из экспоненциального распределения
        val = np.random.exponential(scale=1.0)
        values.append(val)
        # Переход в следующее состояние на основе текущего
        current_state = states[-1]
        next_state = np.random.choice([0, 1, 2], p=P[current_state])
        states.append(next_state)
    
    # Нормируем значения (приводим к диапазону 0-1)
    values = np.array(values)
    values = (values - values.min()) / (values.max() - values.min())
    return states, values

n_steps = 100  # Количество шагов
states1, values1 = generate_markov_chain(P1, n_steps)
states2, values2 = generate_markov_chain(P2, n_steps)

freq1 = np.bincount(states1, minlength=3) / len(states1)
freq2 = np.bincount(states2, minlength=3) / len(states2)

print("\nМатрица P1:")
print(P1)
print("Сгенерированные значения:", values1.tolist())
print("Переходы:", [int(s) for s in states1])
print("Частоты состояний:", freq1.tolist())

print("\nМатрица P2:")
print(P2)
print("Сгенерированные значения:", values2.tolist())
print("Переходы:", [int(s) for s in states2])
print("Частоты состояний:", freq2.tolist())

plt.figure(figsize=(12, 6))
plt.plot(values1, label="Цепь 1 (P1)", alpha=0.7)
plt.plot(values2, label="Цепь 2 (P2)", alpha=0.7)
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
plt.plot(acf1, label="Автокорреляция Цепи 1 (P1)", marker='o')
plt.plot(acf2, label="Автокорреляция Цепи 2 (P2)", marker='o')
plt.title("Автокорреляция цепей Маркова")
plt.xlabel("Лаг")
plt.ylabel("Значение автокорреляции")
plt.legend()
plt.grid(True)
plt.show()
