from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def gerar_topicos(docs: list[str]):
    print(f"Iniciando BERTopic com {len(docs)} documentos...")


    stop_pt = stopwords.words('portuguese')
    stop_en = stopwords.words('english')

    lista_final_stops = stop_pt + stop_en

    vectorizer_model = CountVectorizer(stop_words=lista_final_stops)
    
    topic_model = BERTopic(
        language="multilingual",
        vectorizer_model=vectorizer_model,
        min_topic_size=5, 
        verbose=True
    )

    topics, probs = topic_model.fit_transform(docs)

    info = topic_model.get_topic_info()
    
    print("\nTop 5 Tópicos encontrados:")
    print(info.head())

    return topic_model, info, topics

if __name__ == "__main__":
    from curriculos_loader import carregar_dados_mistos
    
    caminho_csv = "data/curriculos/dataset_bertopic.csv" 
    
    docs, _ = carregar_dados_mistos(caminho_csv)
    modelo, tabela_topicos, lista_topicos = gerar_topicos(docs)
    
    tabela_topicos.to_csv("data/processed/topicos_gerados.csv", index=False)

    df_mapping = pd.DataFrame({
        "Document": docs,
        "Topic": lista_topicos
    })
    df_mapping.to_csv("data/processed/doc_topic_mapping.csv", index=False)
    print("Mapeamento de documentos para tópicos salvo em: data/processed/doc_topic_mapping.csv")