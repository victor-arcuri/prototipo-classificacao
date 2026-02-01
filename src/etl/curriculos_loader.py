import pandas as pd

def carregar_dados_mistos(caminho_csv: str):
    df = pd.read_csv(caminho_csv, quotechar='"')
    
    df['content'] = df['content'].astype(str).fillna('')
    
    df = df[df['content'].str.len() > 15]
    
    docs = df['content'].tolist()
    
    qtd_abstracts = len(df[df['type'] == 'abstract'])
    qtd_artigos = len(df[df['type'] == 'bibliographic_production'])
    print(f"Carregado: {qtd_abstracts} Resumos e {qtd_artigos} Artigos.")
    
    return docs, df

if __name__ == "__main__":
    docs, df = carregar_dados_mistos("data/curriculos/dataset_bertopic.csv")
    print(docs[:3])