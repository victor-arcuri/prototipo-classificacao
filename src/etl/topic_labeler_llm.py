import pandas as pd
import os
import json
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class AreaPrediction(BaseModel):
    """Uma área sugerida com grau de confiança."""
    area_name: str = Field(description="Nome da Grande Área ou Subárea do CNPq (ex: 'Ciência da Computação', 'Saúde Coletiva').")
    confidence: float = Field(description="Grau de certeza (0.0 a 1.0) de que o tópico pertence a esta área.")

class TopicLabel(BaseModel):
    """Estrutura para nomeação e classificação multi-label."""
    short_label: str = Field(
        description="Um nome curto, técnico e acadêmico para o tópico (máx 4 palavras)."
    )
    multi_areas: List[AreaPrediction] = Field(
        description="Lista de até 3 áreas do conhecimento às quais este tópico se conecta, ordenadas por relevância."
    )

def get_labeling_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(TopicLabel)

    system_template = """
    Você é um Classificador Especialista na Taxonomia do CNPq.
    
    SUA MISSÃO:
    1. Dar um nome ACADÊMICO para o tópico.
    2. Classificar as áreas envolvidas.

    --- REGRAS PARA O NOME (RÓTULO) ---
    1. PRIVILEGIE A DISCIPLINA: Não crie nomes apenas somando as variáveis do estudo.
       - RUIM: "Dengue e Transporte" (Isso é o tema do paper, não a área).
       - BOM: "Epidemiologia da Dengue" ou "Saúde Coletiva".
       - RUIM: "Impressão 3D e Saúde".
       - BOM: "Bioengenharia" ou "Tecnologia Assistiva".
    2. GENERALIZE: Se houver artigos variados, encontre o "guarda-chuva" acadêmico que cobre todos.
    3. FORMATO: Curto (máx 4 palavras), Técnico, em Português.

    --- REGRAS PARA AS ÁREAS ---
    1. NÍVEL: Classifique no nível de ÁREA ou SUBÁREA do CNPq (ex: "Genética", "Engenharia Civil", "Educação"). Evite termos vagos como "Tecnologia".
    2. INTERDISCIPLINARIDADE: Se o tópico mistura áreas (ex: Computação aplicada à Biologia), liste ambas com suas respectivas confianças.
    """

    human_template = """
    PALAVRAS-CHAVE DO TÓPICO: {keywords}
    
    TITULOS DOS ARTIGOS (Contexto): {docs}
    
    Gere o rótulo e as áreas:
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])

    return prompt | structured_llm

def processar_topicos(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    chain = get_labeling_chain()
    results = []
    
    print("Iniciando rotulagem ajustada...")
    
    for index, row in df.iterrows():
        topic_id = row['Topic']
        
        is_outlier = (topic_id == -1)
        prefix_msg = "[Outliers]" if is_outlier else f"[Tópico {topic_id}]"

        try:
            print(f"Processando {prefix_msg}...", end="\r")
            
            response: TopicLabel = chain.invoke({
                "keywords": row['Representation'], 
                "docs": row['Representative_Docs']
            })
            
            final_label = response.short_label
            if is_outlier and response.multi_areas[0].confidence < 0.5:
                final_label = "Tópicos Multidisciplinares / Diversos"

            areas_json = [area.model_dump() for area in response.multi_areas]
            
            results.append({
                "Topic": topic_id,
                "LLM_Label": final_label,
                "Multi_Areas_JSON": json.dumps(areas_json),
                "Main_Area": response.multi_areas[0].area_name
            })
            
        except Exception as e:
            print(f"Erro {topic_id}: {e}")
            results.append({
                "Topic": topic_id,
                "LLM_Label": "Erro de Processamento",
                "Multi_Areas_JSON": "[]",
                "Main_Area": "Desconhecido"
            })

    df_labels = pd.DataFrame(results)

    df_final = pd.merge(df, df_labels, on="Topic", how="left")
    
    df_final.to_csv(output_csv, index=False)
    print(f"\nSalvo em {output_csv}")
    
    print(df_final[['Topic', 'LLM_Label', 'Main_Area']].head(5))

if __name__ == "__main__":
    processar_topicos("data/processed/topicos_gerados.csv", "data/processed/topicos_nomeados_llm.csv")