from flask import Flask, render_template, request, jsonify
from chatbot.rag_chatbot import RAGChatbot
from config.settings import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    try:
        response = chatbot.get_response(user_input)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)