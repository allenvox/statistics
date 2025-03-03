import numpy as np
import random
import matplotlib.pyplot as plt
import os

def generate_numbers(size, seed=42):
    np.random.seed(seed)
    numbers = np.arange(1, size + 1)
    return numbers

def sample_with_replacement(numbers, sample_size):
    return [int(random.choice(numbers)) for _ in range(sample_size)]

def sample_without_replacement(numbers, sample_size):
    return list(map(int, random.sample(list(numbers), sample_size)))

def build_matrix(numbers, sample, passes=3):
    n = len(sample) * passes  # Увеличиваем количество строк
    m = len(numbers)
    matrix = np.zeros((n, m), dtype=int)

    # Заполняем матрицу за несколько проходов
    for pass_idx in range(passes):
        for i, value in enumerate(sample):
            idx = np.where(numbers == value)[0][0]
            matrix[pass_idx * len(sample) + i, idx] = 1

    return matrix

def save_matrix_plot(matrix, title, filename):
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, cmap="gray", aspect="auto")
    plt.xlabel("Индексы элементов")
    plt.ylabel("Выборки (несколько проходов)")
    plt.title(title)
    plt.colorbar(label="0 - не выбрано, 1 - выбрано")
    plt.savefig(filename, dpi=300)
    plt.close()

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(str(x) for x in row))

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Основные параметры
sizes = [10, 100]  # Размеры исходных массивов
sample_criteria = [
    ("Кратные пяти", lambda x: x % 5 == 0),
    ("Простые числа", lambda x: all(x % i != 0 for i in range(2, int(np.sqrt(x)) + 1)) and x > 1),
    ("Случайная выборка", lambda x: random.random() > 0.5)
]

for size in sizes:
    numbers = generate_numbers(size)
    print(f"\nИсходный массив (size={size}): {numbers.tolist()}")

    for criterion_name, criterion in sample_criteria:
        sample = [int(x) for x in numbers if criterion(x)]
        print(f"\nВыборка ({len(sample)} элементов) из {size} по критерию: {criterion_name}")
        print(sample)

        if not sample:
            print(f"Для критерия {criterion_name} выборка пуста")
            continue

        # Выборки с возвращением и без
        sample_wr = sample_with_replacement(sample, min(len(sample), 20))
        sample_wor = sample_without_replacement(sample, min(len(sample), 20))

        print("\nПример размещений элементов выборки:")
        print(f"  Размещение с возвращением: {sample_wr}")
        print(f"  Размещение без возвращения: {sample_wor}")

        # Строим матрицы (3 прохода)
        matrix_wr = build_matrix(numbers, sample_wr, passes=3)
        matrix_wor = build_matrix(numbers, sample_wor, passes=3)

        # Выводим часть матрицы в консоль
        print("\nМатрица с возвращением:")
        print_matrix(matrix_wr[:min(10, len(matrix_wr))])  # Выводим первые 10 строк
        print("\nМатрица без возвращения:")
        print_matrix(matrix_wor[:min(10, len(matrix_wor))])

        # Сохраняем графики
        filename_wr = os.path.join(output_dir, f"matrix_WR_{size}_{criterion_name.replace(' ', '_')}.png")
        filename_wor = os.path.join(output_dir, f"matrix_WOR_{size}_{criterion_name.replace(' ', '_')}.png")

        save_matrix_plot(matrix_wr, f"Матрица (с возвращением) size={size}, {criterion_name}", filename_wr)
        save_matrix_plot(matrix_wor, f"Матрица (без возвращения) size={size}, {criterion_name}", filename_wor)

        print(f"Графики сохранены: {filename_wr}, {filename_wor}")
