from flask import Flask, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

@app.route("/converse", methods=["POST"])
def converse():
    data = request.json
    context = data.get("context", "You are a personal companion app for the user and your job is to help the user with their daily needs.")
    user_input = data["question"]

    result = chain.invoke({"context": context, "question": user_input})
    context += f"\nUser: {user_input}\n AI: {result}"

    return jsonify({"response": result, "context": context})

if __name__ == "__main__":
    app.run(debug=True)