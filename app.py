import streamlit as st
import networkx as nx
import pickle
import sys
from pathlib import Path
from pyvis.network import Network
import streamlit.components.v1 as components

ROOT_PATH = Path(__file__).resolve().parent
sys.path.append(str(ROOT_PATH))

BASE_GRAPH_PATH = ROOT_PATH / "data" / "processed" / "grafo_base.gpickle"
FINAL_GRAPH_PATH = ROOT_PATH / "data" / "processed" / "grafo_final.gpickle"
TOPICS_CSV_PATH = ROOT_PATH / "data" / "processed" / "topicos_nomeados_llm.csv"

from src.graph import graph_populate, graph_combiner

st.set_page_config(layout="wide", page_title="Taxonomia Dinâmica UFMG")

st.title("🧬 Taxonomia Viva: CNPq + Lattes")

def get_focused_subgraph(G, selected_categories=None):
    """Retorna apenas os ramos ativos (que têm tópicos do Lattes)."""
    if selected_categories:
        lattes_nodes = [
            n for n, attr in G.nodes(data=True) 
            if attr.get('origin') == 'LATTES' and attr.get('Category') in selected_categories
        ]
    else:
        lattes_nodes = [n for n, attr in G.nodes(data=True) if attr.get('origin') == 'LATTES']
    
    if not lattes_nodes:
        return None 
    
    relevant_nodes = set(lattes_nodes)
    for node in lattes_nodes:
        try:
            parents = list(G.predecessors(node))
            relevant_nodes.update(parents)
            for p in parents:
                ancestors = nx.ancestors(G, p)
                relevant_nodes.update(ancestors)
        except:
            pass
    return G.subgraph(relevant_nodes)

def get_tree_by_area(G, selected_areas):
    """
    Pega o nó da 'Grande Área' (ex: Ciências Exatas) e traz TODOS os seus descendentes,
    mesmo que não tenham tópicos do Lattes conectados.
    """
    nodes_to_keep = set()
    
    for node in G.nodes():
        if node in selected_areas:
            nodes_to_keep.add(node)
            
            descendants = nx.descendants(G, node)
            nodes_to_keep.update(descendants)
            
            ancestors = nx.ancestors(G, node)
            nodes_to_keep.update(ancestors)

    if not nodes_to_keep:
        return None

    return G.subgraph(nodes_to_keep)

def check_and_run_pipeline():
    status_box = st.sidebar.status("Verificando Dados...", expanded=True)
    if not BASE_GRAPH_PATH.exists():
        status_box.write("⚙️ Gerando Grafo Base...")
        graph_populate.build_and_save_cnpq()
    if not TOPICS_CSV_PATH.exists():
        status_box.error("❌ Rode o 'topic_labeler_llm.py' primeiro!")
        st.stop()
    if not FINAL_GRAPH_PATH.exists():
        status_box.write("⚙️ Realizando Enxerto...")
        graph_combiner.run_grafting()
    status_box.update(label="Sistema Pronto", state="complete", expanded=False)


def render_graph():
    if not FINAL_GRAPH_PATH.exists():
        st.warning("Grafo não encontrado.")
        return

    with open(FINAL_GRAPH_PATH, 'rb') as f:
        G_full = pickle.load(f)

    st.sidebar.header("🔍 O que você quer ver?")
    
    all_categories = sorted(list(set(
        [attr.get('Category') for n, attr in G_full.nodes(data=True) if attr.get('Category')]
    )))
    
    selected_areas = st.sidebar.multiselect(
        "Filtrar por Grande Área:",
        options=all_categories,
        default=all_categories
    )
    
    show_full_tree = st.sidebar.checkbox(
        "Mostrar ramos vazios (Dentro da área selecionada)", 
        value=False,
        help="Se marcado, mostra a estrutura inteira das áreas escolhidas acima, mesmo as partes sem tópicos."
    )

    if not selected_areas:
        st.warning("Selecione pelo menos uma área.")
        return

    if show_full_tree:
        G_viz = get_tree_by_area(G_full, selected_areas)
        if G_viz is None:
             st.warning("Não foi possível encontrar a estrutura dessas áreas no grafo.")
             return
        st.sidebar.warning(f"⚠️ Exibindo estrutura completa: {G_viz.number_of_nodes()} nós.")
    else:

        G_viz = get_focused_subgraph(G_full, selected_categories=selected_areas)
        if G_viz is None or G_viz.number_of_nodes() == 0:
            st.warning("Nenhum tópico encontrado para essa seleção.")
            return
        st.sidebar.success(f"Foco: {G_viz.number_of_nodes()} nós relevantes.")

    st.sidebar.divider()

    st.sidebar.header("🎨 Aparência")
    
    layout_mode = st.sidebar.selectbox(
        "Formato do Grafo:",
        ["Hierárquico (Árvore Organizada)", "Explosão (Espalhado)"],
        index=0
    )

    show_labels = st.sidebar.toggle("Mostrar Nomes (Rótulos)", value=True)

    G_plot = G_viz.copy()

    for node in G_plot.nodes():
        attrs = G_plot.nodes[node]
        if 'layer' in attrs:
            attrs['level'] = attrs['layer']
            if attrs.get('origin') == 'LATTES':
                 attrs['level'] = 5 

        if not show_labels:
            if 'title' not in attrs or not attrs['title']:
                attrs['title'] = str(attrs.get('label', node))
            attrs['label'] = " " 
        
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G_plot)

    if layout_mode == "Hierárquico (Árvore Organizada)":
        net.set_options("""
        var options = {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "nodeSpacing": 380, 
              "treeSpacing": 380,
              "levelSeparation": 220
            }
          },
          "physics": { "enabled": false }, 
          "interaction": { "hover": true }
        }
        """)
    else:
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "springLength": 100,
              "springConstant": 0.08,
              "damping": 0.4
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "forceAtlas2Based",
            "stabilization": {
              "enabled": true,
              "iterations": 1000,
              "updateInterval": 25,
              "onlyDynamicEdges": false,
              "fit": true
            }
          },
          "interaction": { "hover": true }
        }
        """)

    path_html = ROOT_PATH / "data" / "processed" / "graph_viz.html"
    net.save_graph(str(path_html))
    with open(path_html, 'r', encoding='utf-8') as f:
        source_code = f.read()
    components.html(source_code, height=760)

if __name__ == "__main__":
    check_and_run_pipeline()
    if st.sidebar.button("🔄 Recarregar Dados"):
        st.rerun()
    render_graph()