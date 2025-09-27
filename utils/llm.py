from langchain_openai import ChatOpenAI

def initialize_llm(api_key, model_name):
    return ChatOpenAI(
        openai_api_key=api_key,
        model_name=model_name,
        temperature=0.7
    )