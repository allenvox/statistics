import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def connection_probability(d, a=None, b=None, mode=1):
    """Возвращает вероятность соединения точек по одной из формул"""
    if mode == 1 and a is not None:  # Первая формула
        return np.exp(-a * d)
    elif mode == 2 and b is not None:  # Вторая формула
        return 1 / (d ** b) if d > 0 else 0
    return 0

def generate_tree(points, a=None, b=None, mixed=False):
    """Генерирует дерево, выбирая корень случайно и соединяя вершины по вероятностям"""
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))
    
    root = np.random.randint(len(points))
    added = {root}
    edges = []

    while len(added) < len(points):
        candidates = list(set(range(len(points))) - added)
        np.random.shuffle(candidates)

        best_connection = None
        max_prob = -1

        for j in candidates:
            possible_connections = list(added)
            np.random.shuffle(possible_connections)

            for i in possible_connections:
                d = distance(points[i], points[j])
                
                if mixed:
                    prob = connection_probability(d, a=a, mode=1)
                else:
                    prob = connection_probability(d, b=b, mode=2)

                if prob > max_prob:
                    max_prob = prob
                    best_connection = (i, j)

        if best_connection:
            edges.append(best_connection)
            added.add(best_connection[1])

    G.add_edges_from(edges)
    return G

def plot_graph(G, points, title):
    """Рисует дерево"""
    plt.figure(figsize=(8, 8))
    pos = {i: points[i] for i in range(len(points))}
    nx.draw(G, pos, with_labels=True, node_size=30, font_size=8, edge_color='gray')
    plt.title(title)
    plt.show()

#np.random.seed(42)
points = np.random.rand(100, 2) * 500

a = 1.0
G = generate_tree(points, a=a, mixed=True)
plot_graph(G, points, f"Дерево, a={a:.1f}")

a = 3.0
G = generate_tree(points, a=a, mixed=True)
plot_graph(G, points, f"Дерево, a={a:.1f}")

b = 1.0
G = generate_tree(points, b=b, mixed=False)
plot_graph(G, points, f"Дерево, b={b:.1f}")

b = 3.0
G = generate_tree(points, b=b, mixed=False)
plot_graph(G, points, f"Дерево, b={b:.1f}")
