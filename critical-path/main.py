# THIS PROGRAM IS BUGGY DEPRECATED SHIT AND DOING WRONG CALCULATIONS DO NOT USE IN COMMERCE

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def calculate_critical_path(edges):
    G = nx.DiGraph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    # Топологическая сортировка
    top_order = list(nx.topological_sort(G))
    
    # Раннее начало и окончание
    early_start = {node: 0 for node in G.nodes()}
    early_finish = {}
    for node in top_order:
        for succ in G.successors(node):
            weight = G[node][succ]['weight']
            early_start[succ] = max(early_start[succ], early_start[node] + weight)
    for node in G.nodes():
        early_finish[node] = early_start[node] + max((G[node][succ]['weight'] for succ in G.successors(node)), default=0)
    
    # Позднее окончание и начало
    last_node = top_order[-1]
    latest_finish = {node: early_finish[last_node] for node in G.nodes()}
    latest_start = {}
    for node in reversed(top_order):
        for pred in G.predecessors(node):
            weight = G[pred][node]['weight']
            latest_finish[pred] = min(latest_finish[pred], latest_finish[node] - weight)
    for node in G.nodes():
        latest_start[node] = latest_finish[node] - max((G[pred][node]['weight'] for pred in G.predecessors(node)), default=0)
    
    # Резервы
    reserves = {}
    local_reserves = {}
    for u, v in G.edges():
        weight = G[u][v]['weight']
        reserves[(u, v)] = latest_start[v] - early_start[u]
        local_reserves[(u, v)] = min((early_start[succ] for succ in G.successors(v)), default=early_finish[v]) - early_finish[u]
    
    # Критический путь
    critical_path = [edge for edge in G.edges() if reserves[edge] == 0]
    critical_cost = sum(G[u][v]['weight'] for u, v in critical_path)
    
    # Вывод таблицы
    data = []
    for u, v in G.edges():
        w = G[u][v]['weight']
        data.append([f"{u}-{v}", w, early_start[u], early_finish[u], latest_start[u], latest_finish[u], reserves[(u, v)], local_reserves[(u, v)]])
    df = pd.DataFrame(data, columns=["Edge", "t_ij", "t_ij_RN", "t_ij_RO", "t_ij_PN", "t_ij_PO", "R", "r"])
    print(df.to_string(index=False))
    print(f"\nCritical Path: {' -> '.join(str(u) for u, v in critical_path)}")
    print(f"Critical Path Cost: {critical_cost}")
    
    # Построение графа
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='black', node_size=2000, font_size=12)
    edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)
    plt.show()

# Данные для двух графов
graph1 = [("A", "B", 2), ("A", "C", 7), ("B", "D", 1), ("C", "D", 3), ("D", "E", 5)]
graph2 = [(1, 2, 4), (1, 3, 6), (2, 4, 7), (2, 5, 5), (3, 5, 2), (4, 6, 3), (5, 6, 1)]

print("Graph 1:")
calculate_critical_path(graph1)
print("\nGraph 2:")
calculate_critical_path(graph2)
