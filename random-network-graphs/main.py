import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_points(num_points, size=100, seed=42):
    """Генерация фиксированного набора точек"""
    np.random.seed(seed)
    x_coords = np.random.uniform(0, size, num_points)
    y_coords = np.random.uniform(0, size, num_points)
    return x_coords, y_coords

def distance(x_coords, y_coords, i, j):
    """Евклидово расстояние"""
    return np.sqrt((x_coords[i] - x_coords[j]) ** 2 + (y_coords[i] - y_coords[j]) ** 2)

def prob_exp(d, a, b):
    """Обновленная формула вероятности P(d) = e^(-a * d^b)"""
    return np.exp(-a * (d ** b))

def prob_inverse(d, b):
    """Функция вероятности P(d) = 1 / d^b, нормированная"""
    return min(1, 10 / (d ** b)) if d > 1 else 1  

def build_tree(num_points, method, param_a, param_b=1, max_degree=5, size=100, seed=42):
    """Формируем дерево с ограничением максимальной степени вершин"""
    x_coords, y_coords = generate_points(num_points, size, seed)
    G = nx.Graph()
    
    # Соединяем первую точку как корень
    G.add_node(0)
    connected_nodes = {0}  

    for j in range(1, num_points):
        best_i = None
        max_p = 0

        # Ищем, с кем соединить новую точку
        for i in connected_nodes:
            if G.degree(i) >= max_degree:  # Если вершина уже имеет max_degree связей — пропускаем
                continue
            
            d = distance(x_coords, y_coords, i, j)
            if 2 <= d <= 40:  # Ограничиваем дальность связи
                if method == "a":
                    p = prob_exp(d, param_a, param_b)
                else:
                    p = prob_inverse(d, param_b)
                    
                if p > max_p and np.random.rand() < p:
                    max_p = p
                    best_i = i

        # Если нашли с кем соединить, добавляем ребро
        if best_i is not None:
            G.add_edge(best_i, j)
        else:
            # В крайнем случае соединяем с ближайшей точкой, если у нее еще есть "свободные слоты"
            closest = min(
                [i for i in connected_nodes if G.degree(i) < max_degree], 
                key=lambda i: distance(x_coords, y_coords, i, j),
                default=None
            )
            if closest is not None:
                G.add_edge(closest, j)

        connected_nodes.add(j)  

    return G, x_coords, y_coords

def plot_tree(G, x_coords, y_coords, title, filename):
    """Рисуем и сохраняем дерево, выделяя корень"""
    plt.figure(figsize=(12, 12))
    pos = {i: (x_coords[i], y_coords[i]) for i in range(len(x_coords))}
    
    # Рисуем ребра и вершины
    nx.draw(G, pos, with_labels=False, node_size=50, edge_color="gray")
    
    # Выделяем корневую вершину красным
    plt.scatter(x_coords[0], y_coords[0], color="red", s=100, label="Root Node")
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.close()

# Генерируем деревья с разными ограничениями степени вершин
num_points = 300
seed_value = 666

for a in [0.1, 0.5, 1, 2, 3]:
    for b in [0.5, 0.8, 1, 2, 3]:
        for max_deg in [3, 5, 8]:  # Тестируем разные максимальные степени вершин
            G, x_coords, y_coords = build_tree(num_points, method="a", param_a=a, param_b=b, max_degree=max_deg, seed=seed_value)
            filename = f"f1_{a}-{b}-deg{max_deg}.png"
            plot_tree(G, x_coords, y_coords, f"Дерево при a={a}, b={b}, max_deg={max_deg}", filename)

for b in [0.5, 0.8, 1, 2, 3]:
    for max_deg in [3, 5, 8]:
        G, x_coords, y_coords = build_tree(num_points, method="b", param_a=1, param_b=b, max_degree=max_deg, seed=seed_value)
        filename = f"f2_{b}-deg{max_deg}.png"
        plot_tree(G, x_coords, y_coords, f"Дерево при b={b}, max_deg={max_deg}", filename)
