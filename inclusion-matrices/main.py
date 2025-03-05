import numpy as np
import random
import matplotlib.pyplot as plt
import os

def generate_numbers(size):
    return np.arange(1, size + 1)

def sample_with_replacement(numbers, sample_size, seed=None):
    if seed is not None:
        random.seed(seed)
    return [int(random.choice(numbers)) for _ in range(sample_size)]

def sample_without_replacement(numbers, sample_size, seed=None):
    if seed is not None:
        random.seed(seed)
    return list(map(int, random.sample(list(numbers), sample_size)))

def build_frequency_matrix(numbers, sample, passes=3, use_seed=True):
    n = len(sample)  
    m = len(numbers)  
    matrix = np.zeros((n, m), dtype=int)

    for pass_idx in range(passes):
        seed = random.randint(0, 10000) if use_seed else None
        sample_wr = sample_with_replacement(sample, len(sample), seed)

        for i, value in enumerate(sample_wr):
            idx = np.where(numbers == value)[0][0]  
            matrix[i, idx] += 1  

    return matrix

def save_matrix_plot(matrix, title, filename):
    plt.figure(figsize=(12, 6))
    plt.imshow(matrix, cmap="plasma", aspect="auto")
    plt.xlabel("Индексы элементов")
    plt.ylabel("Выборки")
    plt.title(title)
    plt.colorbar(label="Частота появления элемента")
    plt.savefig(filename, dpi=300)
    plt.close()

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{x:2}" for x in row))

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

sizes = [10, 100]  
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

        sample_wr = sample_with_replacement(sample, min(len(sample), 20))
        sample_wor = sample_without_replacement(sample, min(len(sample), 20))

        print("\nПример размещений элементов выборки:")
        print(f"  Размещение с возвращением: {sample_wr}")
        print(f"  Размещение без возвращения: {sample_wor}")

        matrix_wr = build_frequency_matrix(numbers, sample_wr, passes=13, use_seed=True)
        matrix_wor = build_frequency_matrix(numbers, sample_wor, passes=42, use_seed=True)

        print("\nМатрица частот с возвращением, 13 проходов:")
        print_matrix(matrix_wr)

        print("\nМатрица частот без возвращения, 42 прохода:")
        print_matrix(matrix_wor)

        filename_wr = os.path.join(output_dir, f"freq_matrix_WR_{size}_{criterion_name.replace(' ', '_')}.png")
        filename_wor = os.path.join(output_dir, f"freq_matrix_WOR_{size}_{criterion_name.replace(' ', '_')}.png")

        save_matrix_plot(matrix_wr, f"Частотная матрица (с возвращением) size={size}, {criterion_name}", filename_wr)
        save_matrix_plot(matrix_wor, f"Частотная матрица (без возвращения) size={size}, {criterion_name}", filename_wor)

        print(f"Графики сохранены: {filename_wr}, {filename_wor}")
