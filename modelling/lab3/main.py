import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_points(num_points, size=100):
    """Генерирует случайные точки в квадрате size x size"""
    x_coords = np.random.uniform(0, size, num_points)
    y_coords = np.random.uniform(0, size, num_points)
    
    # Сортируем точки по x (слева направо)
    indices = np.argsort(x_coords)
    return x_coords[indices], y_coords[indices]

def distance(x_coords, y_coords, i, j):
    """Вычисляет евклидово расстояние между точками i и j"""
    return np.sqrt((x_coords[i] - x_coords[j]) ** 2 + (y_coords[i] - y_coords[j]) ** 2)

def prob_exp(d, a):
    """Формула вероятности e^(-a * d)"""
    return np.exp(-a * d)

def prob_inverse(d, b):
    """Формула вероятности 1 / d^b"""
    return 1 / (d ** b) if d > 1 else 1  # Учитываем минимальное расстояние

def build_tree(num_points, method, param, size=100):
    """Строит дерево в зависимости от метода (a или b)"""
    x_coords, y_coords = generate_points(num_points, size)
    G = nx.DiGraph()
    G.add_nodes_from(range(num_points))
    
    connected = set()  # Множество уже подключённых вершин
    connected.add(0)  # Корень — самая левая точка

    for j in range(1, num_points):  # Подключаем каждую новую точку
        best_parent = None
        best_prob = 0
        
        for i in range(j):  # Ищем родителя среди предыдущих точек
            d = distance(x_coords, y_coords, i, j)
            if 1 <= d <= 10:  # Ограничение по расстоянию
                if method == "a":
                    p = prob_exp(d, param)  # Используем только e^(-a*d)
                elif method == "b":
                    p = prob_inverse(d, param)  # Используем только 1 / d^b
                
                if p > best_prob:
                    best_parent = i
                    best_prob = p
        
        # Добавляем ребро, если нашёлся родитель
        if best_parent is not None:
            G.add_edge(best_parent, j)
            connected.add(j)
    
    # 🔥 Новый блок: Принудительно объединяем изолированные точки в одно дерево
    components = list(nx.weakly_connected_components(G))
    if len(components) > 1:  # Если граф несвязный
        print(f"Обнаружены {len(components)} отдельных деревьев. Объединяем их...")
        main_component = components[0]  # Берём первую компоненту как основу
        for component in components[1:]:
            nearest = min(main_component, key=lambda i: min(distance(x_coords, y_coords, i, j) for j in component))
            farthest = min(component, key=lambda j: distance(x_coords, y_coords, nearest, j))
            G.add_edge(nearest, farthest)
            main_component |= component  # Объединяем множества

    return G, x_coords, y_coords

def plot_tree(G, x_coords, y_coords, title):
    """Рисует дерево с заголовком"""
    plt.figure(figsize=(12, 12))
    pos = {i: (x_coords[i], y_coords[i]) for i in range(len(x_coords))}
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=6, edge_color="gray", arrows=True)
    plt.suptitle(title, fontsize=14, fontweight='bold')  # Принудительный заголовок
    plt.show()

# Запуск с разными параметрами
num_points = 300  # Можно менять на 500 или другое значение

# Вариация параметра a
for a in [0.1, 0.5, 1, 2, 5]:  # От "сильных связей" до "разреженных"
    G, x_coords, y_coords = build_tree(num_points, method="a", param=a)
    plot_tree(G, x_coords, y_coords, f"Дерево при a = {a}")

# Вариация параметра b
for b in [1, 2, 3, 4]:  # Чем больше b, тем разреженнее дерево
    G, x_coords, y_coords = build_tree(num_points, method="b", param=b)
    plot_tree(G, x_coords, y_coords, f"Дерево при b = {b}")
