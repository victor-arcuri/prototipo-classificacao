import networkx as nx
import json
from pathlib import Path
import pickle

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
TAXONOMY_PATH = ROOT_PATH / "data" / "taxonomies" / "cnpq_taxonomy.json"
OUTPUT_BASE_GRAPH_PATH = ROOT_PATH / "data" / "processed" / "grafo_base.gpickle"

def get_level(level: int) -> str:
    if (level == 0):
        return "Grande Área"
    elif (level == 1):
        return "Área"
    elif (level == 2):
        return "Subárea"
    elif (level == 3):
        return "Especialidade"
    else:
        return "Desconhecido"

def recursive_graph_populate(graph: nx.DiGraph, starting_node, level: int = 0) -> str:
    name = starting_node["name"]
    formated_level = get_level(level)
    size = 25 - (level * 5)
    graph.add_node(
        name, 
        label=name,
        title=f"Fonte: CNPq ({formated_level})",
        origin="CNPQ", 
        level=formated_level, 
        layer=level,
        color="#97C2FC",
        size=size
    )
    for child in starting_node.get("children", []):
        child_name = recursive_graph_populate(graph, child, level + 1)
        if child_name: 
            graph.add_edge(name, child_name, relation="subarea_de")
    return name

def build_and_save_cnpq():
    print("Construindo grafo base do CNPq...")
    
    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {TAXONOMY_PATH}")

    taxonomy_data = json.loads(TAXONOMY_PATH.read_text(encoding='utf-8'))
    G = nx.DiGraph(name="Taxonomia CNPq Base")
    
    if isinstance(taxonomy_data, dict):
        recursive_graph_populate(G, taxonomy_data)
    elif isinstance(taxonomy_data, list):
        G.add_node("CNPQ_Raiz", label="CNPq", color="#000000", size=30, origin="CNPQ")
        for item in taxonomy_data:
            child = recursive_graph_populate(G, item, level=0)
            G.add_edge("CNPQ_Raiz", child)

    with open(OUTPUT_BASE_GRAPH_PATH, 'wb') as f:
        pickle.dump(G, f)
    
    print(f"Sucesso! Grafo base salvo em: {OUTPUT_BASE_GRAPH_PATH}")
    print(f"   Nós: {G.number_of_nodes()} | Arestas: {G.number_of_edges()}")

if __name__ == "__main__":
    build_and_save_cnpq()