import numpy as np
import random
import matplotlib.pyplot as plt

def generate_numbers(size, seed=42):
    """Генерация массива чисел от 1 до size."""
    np.random.seed(seed)
    numbers = np.arange(1, size + 1)
    return numbers

def sample_with_replacement(numbers, sample_size):
    """Выборка с возвращением"""
    return [int(random.choice(numbers)) for _ in range(sample_size)]  # Приводим к int

def sample_without_replacement(numbers, sample_size):
    """Выборка без возвращения"""
    return list(map(int, random.sample(list(numbers), sample_size)))  # Приводим к int

def build_matrix(numbers, sample):
    """Формирование матрицы включения"""
    n = len(sample)
    m = len(numbers)
    matrix = np.zeros((n, m), dtype=int)

    # Заполняем матрицу: 1 если элемент входит в выборку
    for i, value in enumerate(sample):
        idx = np.where(numbers == value)[0][0]  # Индекс в исходном массиве
        matrix[i, idx] = 1

    return matrix

def plot_matrix(matrix, title):
    """Визуализация матрицы"""
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, cmap="gray", aspect="auto")
    plt.xlabel("Индексы элементов")
    plt.ylabel("Выборки")
    plt.title(title)
    plt.colorbar(label="0 - не выбрано, 1 - выбрано")
    plt.show()

# Основные параметры
sizes = [10, 100, 1000]  # Размер исходных массивов
sample_criteria = [
    ("Кратные пяти", lambda x: x % 5 == 0),
    ("Простые числа", lambda x: all(x % i != 0 for i in range(2, int(np.sqrt(x)) + 1)) and x > 1),
    ("Случайная выборка", lambda x: random.random() > 0.5)
]

for size in sizes:
    numbers = generate_numbers(size)
    print(f"\nИсходный массив (size={size}):", numbers.tolist())  # Выводим как список обычных чисел

    for criterion_name, criterion in sample_criteria:
        sample = [int(x) for x in numbers if criterion(x)]  # Приводим все числа к int
        print(f"\nВыборка ({len(sample)} элементов) из {size} по критерию: {criterion_name}")
        print(sample)

        # Выборка с возвращением и без
        sample_wr = sample_with_replacement(sample, min(len(sample), 20))
        sample_wor = sample_without_replacement(sample, min(len(sample), 20))

        # Формируем матрицы
        matrix_wr = build_matrix(numbers, sample_wr)
        matrix_wor = build_matrix(numbers, sample_wor)

        # Визуализируем матрицы
        plot_matrix(matrix_wr, f"Матрица (с возвращением) для size={size}, {criterion_name}")
        plot_matrix(matrix_wor, f"Матрица (без возвращения) для size={size}, {criterion_name}")
