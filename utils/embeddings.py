from langchain_openai import OpenAIEmbeddings

def initialize_embeddings(api_key, model_name):
    return OpenAIEmbeddings(
        openai_api_key=api_key,
        model=model_name
    )