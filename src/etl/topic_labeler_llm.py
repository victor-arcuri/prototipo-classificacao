import pandas as pd
import os
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class TopicLabel(BaseModel):
    """Estrutura para nomeação de tópicos científicos."""
    short_label: str = Field(
        description="Um nome curto, técnico e acadêmico para a área de pesquisa (3-6 palavras). Ex: 'Epidemiologia Computacional'"
    )
    category: str = Field(
        description="A grande área do conhecimento a qual esse tópico pertence. Ex: 'Ciência da Computação', 'Saúde Pública', 'Engenharia'"
    )
    confidence_score: float = Field(
        description="Um nível de confiança (0.0 a 1.0) de que o nome reflete bem os documentos fornecidos."
    )

def get_labeling_chain():

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    

    structured_llm = llm.with_structured_output(TopicLabel)

    system_template = """
    Você é um Classificador de Áreas do Conhecimento (Taxonomia CNPq).
    Sua missão é identificar a DISCIPLINA ou SUBÁREA TÉCNICA, não resumir o conteúdo dos artigos.

    REGRAS DE OURO:
    1. GENERALIZE: Se os artigos falam de "Robótica para evitar violência" e "Robótica para ensino", o rótulo deve ser APENAS "Robótica Educacional". Corte os detalhes específicos de aplicação.
    2. CURTO E DIRETO: Use no máximo 4 palavras.
    3. PADRÃO ACADÊMICO: Use termos de disciplinas (ex: "Epidemiologia", "Engenharia de Software", "Sistemas Distribuídos").
    4. EVITE CONJUNÇÕES LONGAS: Não use "Estudo sobre...", "Análise de...", "Correlação entre...". Vá direto ao tema.
    5. IDIOMA: Sempre em Português.
    """

    human_template = """
    Analise este Tópico:
    
    PALAVRAS-CHAVE (Keywords):
    {keywords}
    
    DOCUMENTOS REPRESENTATIVOS (Amostra):
    {docs}
    
    Gere o rótulo acadêmico estruturado para este tópico.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])

    chain = prompt | structured_llm
    return chain

def processar_topicos(input_csv: str, output_csv: str):
    print(f"Lendo dados de: {input_csv}")
    df = pd.read_csv(input_csv)
    
    chain = get_labeling_chain()
    
    results = []
    
    print("Iniciando rotulagem com IA...")
    
    for index, row in df.iterrows():
        topic_id = row['Topic']
        
        if topic_id == -1:
            results.append({
                "Topic": topic_id,
                "LLM_Label": "Outros / Multidisciplinar",
                "Category": "Geral",
                "Confidence": 0.0
            })
            continue

        keywords = row['Representation']
        docs = row['Representative_Docs']
        
        try:
            print(f"Processando Tópico {topic_id}...", end="\r")
            response: TopicLabel = chain.invoke({"keywords": keywords, "docs": docs})
            
            results.append({
                "Topic": topic_id,
                "LLM_Label": response.short_label,
                "Category": response.category,
                "Confidence": response.confidence_score
            })
            
        except Exception as e:
            print(f"Erro no tópico {topic_id}: {e}")
            results.append({
                "Topic": topic_id,
                "LLM_Label": f"Erro: {row['Name']}", 
                "Category": "Erro",
                "Confidence": 0.0
            })

    df_labels = pd.DataFrame(results)
    
    df_final = pd.merge(df, df_labels, on="Topic")
    
    df_final.to_csv(output_csv, index=False)
    print(f"\n\nSucesso! Arquivo salvo em: {output_csv}")
    
    print(df_final[['Topic', 'LLM_Label', 'Category']].head(10))

if __name__ == "__main__":
    INPUT_FILE = "data/processed/topicos_gerados.csv"
    OUTPUT_FILE = "data/processed/topicos_nomeados_llm.csv"
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERRO: Configure a variável de ambiente OPENAI_API_KEY")
    else:
        processar_topicos(INPUT_FILE, OUTPUT_FILE)