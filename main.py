import networkx as nx
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Função para ler o arquivo de políticos e suas votações.
def read_politicians_file(filename):
    politicians = {}
    file_path = os.path.join("datasets", filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = line.strip().split(';')
            name = data[0]
            party = data[1]
            votes = int(data[2])
            politicians[name] = {'party': party, 'votes': votes}
    return politicians

# Função para ler o arquivo de grafo e criar um grafo com base nas informações.
def create_graph_from_file(filename):
    G = nx.Graph()
    file_path = os.path.join("datasets", filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = line.strip().split(';')
            politician_a = data[0]
            politician_b = data[1]
            weight = int(data[2])
            G.add_edge(politician_a, politician_b, weight=weight)
    return G

# Função para realizar análises específicas no grafo.
def analyze_graph(graph):
    # Exemplo de análise: obter a lista de arestas com seus pesos.
    edge_weights = [(edge[0], edge[1], graph[edge[0]][edge[1]]['weight']) for edge in graph.edges()]
    return edge_weights

def normalize_graph_weights(graph):
    normalized_graph = graph.copy()
    
    for u, v, data in normalized_graph.edges(data=True):
        weight = data['weight']
        min_votes = min(normalized_graph.nodes[u]['votes'], normalized_graph.nodes[v]['votes'])
        normalized_weight = weight / min_votes
        normalized_graph[u][v]['weight'] = normalized_weight
    
    return normalized_graph

def apply_threshold(graph, threshold):
    thresholded_graph = graph.copy()

    edges_to_remove = []
    for u, v, data in thresholded_graph.edges(data=True):
        weight = data['weight']
        if weight < threshold:
            edges_to_remove.append((u, v))

    for u, v in edges_to_remove:
        thresholded_graph.remove_edge(u, v)

    return thresholded_graph

# Lista de anos disponíveis
available_years = list(range(2001, 2024))

# Solicitar ao usuário o ano e os partidos
year = int(input("Digite o ano a ser analisado: "))
parties = input("Digite os partidos a serem analisados (separados por vírgula): ").split(',')
threshold = float(input("Digite o valor do threshold: "))

if year not in available_years:
    print("Ano inválido.")
else:
    politicians_file = f'politicians{year}.txt'
    graph_file = f'graph{year}.txt'

    politicians_data = read_politicians_file(politicians_file)
    graph = create_graph_from_file(graph_file)

    # Filtrar os políticos pelo partido
    filtered_politicians = {name: data for name, data in politicians_data.items() if data['party'] in parties}

    # Filtrar o grafo pelas arestas dos políticos do partido escolhido
    filtered_graph = graph.subgraph(filtered_politicians.keys())

    # Adicionar informações sobre o número de votações ao grafo filtrado
    for node in filtered_graph.nodes():
        filtered_graph.nodes[node]['votes'] = filtered_politicians[node]['votes']

    # Normalizar os pesos das arestas do grafo filtrado
    normalized_graph = normalize_graph_weights(filtered_graph)

    # Aplicar o threshold no grafo normalizado
    thresholded_graph = apply_threshold(normalized_graph, threshold)

    # Salvar o grafo filtrado, normalizado e thresholded em um arquivo de texto na pasta "grafosComFiltro"
    graph_filename = f'consulta{len(os.listdir("grafosComFiltro"))}.txt'
    graph_path = os.path.join("grafosComFiltro", graph_filename)
    nx.write_weighted_edgelist(thresholded_graph, graph_path, delimiter=';')

    print(f"Grafo {graph_filename} gerado com sucesso na pasta 'grafosComFiltro'.")

# Plotar e salvar o grafo usando draw_spring
plt.figure(figsize=(10, 8))

# Criar um subgrafo contendo apenas os nós relevantes após o threshold
filtered_thresholded_graph = thresholded_graph.subgraph(thresholded_graph.nodes())

# Gerar uma paleta de cores baseada no número de partidos
num_parties = len(parties)
palette = sns.color_palette("husl", n_colors=num_parties)

# Criar um dicionário para mapear partido a cor
party_colors = {party: palette[i] for i, party in enumerate(parties)}

# Filtrar os políticos pelo partido após o threshold
filtered_politicians_after_threshold = {node: data for node, data in filtered_politicians.items() if node in filtered_thresholded_graph.nodes()}

node_colors = [party_colors[filtered_politicians_after_threshold[node]['party']] for node in filtered_thresholded_graph.nodes()]

nx.draw(filtered_thresholded_graph, with_labels=True, font_size=10, node_size=500, node_color=node_colors, edge_color='gray')

# Adicionar legenda das cores
legend_labels = [f"{party}: {party_colors[party]}" for party in parties]
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=party_colors[party], markersize=10, label=label) for party, label in zip(parties, legend_labels)], loc='upper left')

plt.title(f"Grafo Filtrado e Thresholded - Ano {year} - Partidos: {', '.join(parties)}")
plot_filename = f'consulta{len(os.listdir("grafosComFiltro_Plotados"))}.png'
plot_path = os.path.join("grafosComFiltro_Plotados", plot_filename)
plt.savefig(plot_path)
plt.close()
print(f"Grafo plotado e salvo como '{plot_filename}' na pasta 'grafosComFiltro_Plotados'.")