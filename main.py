from chatbot.rag_chatbot import RAGChatbot
from config.settings import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT

def main():
    # Initialize the chatbot
    chatbot = RAGChatbot(
        openai_api_key=OPENAI_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        pinecone_environment=PINECONE_ENVIRONMENT,
        index_name="restaurant",
        chat_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        namespace="Recipies"
    )

    print("Welcome to the Restaurant Recipe Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chatbot.get_response(user_input)
        print(f"Bot: {response}")

if __name__ == '__main__':
    main()