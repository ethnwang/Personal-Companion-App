from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def converse():
    context = "You are a personal companion app for the user and your job is to help the user with their daily needs."
    print("Welcome to your personal companion app! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        result = chain.invoke({"context": context, "question": user_input})
        print("Personal Companion: ", result)
        context += f"\nUser: {user_input}\n AI: {result}"

if __name__ == "__main__":
    converse()