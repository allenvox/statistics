import numpy as np
import matplotlib.pyplot as plt

# Заданные параметры
k = 0.6115  # Найденный коэффициент k
sqrt3_4 = np.sqrt(3) / 4  # Значение F(pi/6)

# Обратная функция для генерации случайных величин
def inverse_transform_sampling(u):
    if u <= sqrt3_4:
        # Для первой части функции распределения
        return np.arcsin(2 * u) - np.pi / 6
    else:
        # Для второй части функции распределения
        return np.pi / 6 - np.log(1 - (u - sqrt3_4) / k)

# Генерация случайных величин
np.random.seed(42)  # Для воспроизводимости результатов
u_samples = np.random.uniform(0, 1, 10000)  # Генерация 10000 случайных чисел U[0, 1]
x_samples = np.array([inverse_transform_sampling(u) for u in u_samples])  # Применение обратной функции

# Построение гистограммы сгенерированных данных
plt.hist(x_samples, bins=50, density=True, alpha=0.6, color='g', label='Сгенерированные данные')

# Построение графика исходной плотности распределения f(x)
x_values = np.linspace(-np.pi/6, np.pi, 1000)  # Точки для построения графика f(x)
f_values = np.where(
    x_values <= np.pi/6,
    0.5 * np.cos(x_values + np.pi/6),  # Первая часть f(x)
    k * np.exp(-(x_values - np.pi/6))  # Вторая часть f(x)
)

plt.plot(x_values, f_values, 'r-', label='Исходная плотность f(x)')
plt.title('Тестовые данные на посчитанной обратной функции / исходная плотность')
plt.xlabel('x')
plt.ylabel('f (x)')
plt.legend()
plt.grid(True)
plt.show()
