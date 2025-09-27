from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeLangChain

def initialize_pinecone(api_key, environment, index_name, embeddings, namespace="Recipies"):
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists, if not, create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # Dimension for text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=environment)
        )
    
    # Initialize LangChain Pinecone vector store with namespace
    vector_store = PineconeLangChain.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    return vector_store