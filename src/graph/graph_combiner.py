import networkx as nx
import pandas as pd
import pickle
import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
INPUT_BASE_GRAPH_PATH = ROOT_PATH / "data" / "processed" / "grafo_base.gpickle"
TOPICS_PATH = ROOT_PATH / "data" / "processed" / "topicos_nomeados_llm.csv"
OUTPUT_FINAL_GRAPH_PATH = ROOT_PATH / "data" / "processed" / "grafo_final.gpickle"

def graft_lattes_topics(G: nx.DiGraph, df_topics: pd.DataFrame):
    print("Carregando modelo de Embeddings...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    cnpq_leaves = [n for n in G.nodes() 
                   if G.nodes[n].get('origin') == 'CNPQ' and G.out_degree(n) == 0]
    
    print(f"Calculando vetores para {len(cnpq_leaves)} nós FOLHA (Nível mais baixo)...")

    cnpq_embeddings = model.encode(cnpq_leaves)


    print("Iniciando enxerto multi-áreas...")
    count_nodes = 0
    count_edges = 0
    
    for _, row in df_topics.iterrows():
        if row['Topic'] == -1: 
            continue

        label = row['LLM_Label']
        
        try:
            areas = json.loads(row['Multi_Areas_JSON'])
        except Exception as e:
            print(f"Erro lendo JSON do tópico {row['Topic']}: {e}")
            continue

        main_category = areas[0]['area_name'] if areas else "Indefinido"
        
        micro_graph_file = f"data/processed/micro_grafos/topico_{row['Topic']}.gpickle"
        
        if not G.has_node(label):
            G.add_node(
                label, 
                label=label,
                Category=main_category, 
                origin="LATTES",
                color="#FF5733",
                size=15, 
                layer=5,
                micro_path=micro_graph_file
            )
            count_nodes += 1

        for area in areas:
            area_name = area['area_name']
            confidence = area['confidence']

            if confidence < 0.6:
                continue

            search_query = f"{label} ({area_name})"
            query_embedding = model.encode([search_query])

            similarities = cosine_similarity(query_embedding, cnpq_embeddings)[0]
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            parent_node_name = cnpq_leaves[best_idx]

            if parent_node_name.strip().lower() == label.strip().lower():
                continue

            if G.has_edge(parent_node_name, label):
                continue

            if best_score > 0.4:
                G.add_edge(
                    parent_node_name, 
                    label, 
                    weight=float(best_score), 
                    relation="interdisciplinar",
                    context=area_name
                )
                
                print(f"  {label} --> {parent_node_name} [Ctx: {area_name}] (Score: {best_score:.2f})")
                count_edges += 1

    print(f"Grafo Final Concluído!")
    print(f"Tópicos inseridos: {count_nodes}")
    print(f"Conexões interdisciplinares criadas: {count_edges}")
    return G

def run_grafting():
    if not INPUT_BASE_GRAPH_PATH.exists():
        print("Erro: Rode o script 'graph_cnpq.py' primeiro!")
        return

    print("Carregando grafo base...")
    with open(INPUT_BASE_GRAPH_PATH, 'rb') as f:
        G = pickle.load(f)

    if not TOPICS_PATH.exists():
        print("Erro: CSV de tópicos não encontrado!")
        return
    
    df_topics = pd.read_csv(TOPICS_PATH)
    
    G_final = graft_lattes_topics(G, df_topics)
    
    with open(OUTPUT_FINAL_GRAPH_PATH, 'wb') as f:
        pickle.dump(G_final, f)
    
    print(f"Grafo final salvo em: {OUTPUT_FINAL_GRAPH_PATH}")

if __name__ == "__main__":
    run_grafting()