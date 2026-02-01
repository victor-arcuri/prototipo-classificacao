import networkx as nx
import pandas as pd
import pickle
import numpy as np
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
    
    cnpq_nodes = [n for n, attr in G.nodes(data=True) if attr.get('origin') == 'CNPQ']
    
    print(f"Calculando vetores para {len(cnpq_nodes)} nós da taxonomia oficial...")
    cnpq_embeddings = model.encode(cnpq_nodes)

    print("Iniciando enxerto dos tópicos descobertos...")
    count = 0
    
    for _, row in df_topics.iterrows():
        label = row['LLM_Label']
        category = row['Category']
        confidence = row['Confidence']
        
        if row['Topic'] == -1 or confidence < 0.5: 
            continue

        search_query = f"{label} ({category})"
        query_embedding = model.encode([search_query])

        similarities = cosine_similarity(query_embedding, cnpq_embeddings)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        parent_node_name = cnpq_nodes[best_idx]

        G.add_node(
            label, 
            label=label,
            Category=category, 
            title=f"<b>{label}</b><br>Grande Área: {category}<br>Confiança IA: {confidence}",
            origin="LATTES",
            color="#FF5733",
            size=15, 
            layer=5
        )

        G.add_edge(parent_node_name, label, weight=float(best_score), relation="descoberto_em")
        
        print(f"  {label} --> {parent_node_name} (Score: {best_score:.2f})")
        count += 1

    print(f"Grafo Final Concluído! {count} novos tópicos inseridos.")
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