import pandas as pd
import networkx as nx
import pickle
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
MAPPING_PATH = ROOT_PATH / "data" / "processed" / "doc_topic_mapping.csv"
MICRO_GRAPHS_DIR = ROOT_PATH / "data" / "processed" / "micro_grafos"

MICRO_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

class Relation(BaseModel):
    """Uma relação extraída do texto entre duas entidades."""
    source: str = Field(description="Nó de origem (ex: Nome do Pesquisador, Tecnologia, Algoritmo).")
    target: str = Field(description="Nó de destino (ex: Área de Aplicação, Doença, Ferramenta).")
    relation_type: str = Field(description="Verbo ou ação que os conecta (ex: 'desenvolve', 'aplica-se em', 'investiga').")

class GraphExtraction(BaseModel):
    """Lista de relações extraídas de um documento."""
    triples: List[Relation] = Field(description="Lista de tríades extraídas do texto.")

def get_extraction_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(GraphExtraction)

    system_template = """
    Você é um especialista em extração de informações acadêmicas e grafos de conhecimento.
    Sua tarefa é ler um trecho de currículo Lattes (ou resumo de artigo) e extrair as relações diretas de conhecimento.
    
    REGRA DE OURO:
    - Extraia entidades limpas e curtas.
    - O relacionamento deve ser um verbo ou ação clara no presente ou infinitivo.
    - Foque em tecnologias, métodos, objetos de estudo e aplicações.
    """

    human_template = """
    TEXTO ACADÊMICO: {text}
    
    Extraia as tríades de conhecimento:
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])

    return prompt | structured_llm

def build_micro_graphs():
    if not MAPPING_PATH.exists():
        print("Erro: Arquivo doc_topic_mapping.csv não encontrado. Rode o topic_modeling.py primeiro.")
        return

    df = pd.read_csv(MAPPING_PATH)
    chain = get_extraction_chain()
    
    grouped = df.groupby('Topic')
    
    for topic_id, group in grouped:
        if topic_id == -1:
            continue
            
        print(f"\nConstruindo Micro-Grafo para o Tópico {topic_id} ({len(group)} documentos)...")
        
        G_micro = nx.DiGraph(name=f"Micro_Grafo_Topico_{topic_id}")
        
        docs_series = group['Document'].apply(str)
        
        docs_limpos = docs_series.str.strip().str.replace(r'\s+', ' ', regex=True)

        mascara_unicos = ~docs_limpos.str.lower().duplicated()
        docs_unicos = docs_limpos[mascara_unicos]

        docs_to_process = docs_unicos.sort_values(key=lambda x: x.str.len(), ascending=False).head(5)
        
        for doc in docs_to_process:
            try:
                extraction: GraphExtraction = chain.invoke({"text": doc})
                
                for triple in extraction.triples:
                    G_micro.add_node(triple.source, type="Entity")
                    G_micro.add_node(triple.target, type="Entity")
                    G_micro.add_edge(triple.source, triple.target, relation=triple.relation_type)
                    
            except Exception as e:
                print(f"Erro ao extrair de um documento no tópico {topic_id}: {e}")
                
        output_path = MICRO_GRAPHS_DIR / f"topico_{topic_id}.gpickle"
        with open(output_path, 'wb') as f:
            pickle.dump(G_micro, f)
            
        print(f" -> Salvo: {output_path.name} (Nós: {G_micro.number_of_nodes()}, Arestas: {G_micro.number_of_edges()})")

if __name__ == "__main__":
    build_micro_graphs()