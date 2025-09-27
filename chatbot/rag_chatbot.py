from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils.llm import initialize_llm
from utils.embeddings import initialize_embeddings
from utils.pinecone_utils import initialize_pinecone

class RAGChatbot:
    def __init__(self, openai_api_key, pinecone_api_key, pinecone_environment, index_name, chat_model, embedding_model, namespace="Recipies"):
        # Initialize embeddings
        self.embeddings = initialize_embeddings(openai_api_key, embedding_model)
        
        # Initialize Pinecone vector store with namespace
        self.vector_store = initialize_pinecone(pinecone_api_key, pinecone_environment, index_name, self.embeddings, namespace)
        
        # Initialize LLM
        self.llm = initialize_llm(openai_api_key, chat_model)
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize the conversational retrieval chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory
        )
    
    def get_response(self, query):
        # Get response from the chain
        result = self.chain({"question": query})
        return result["answer"]