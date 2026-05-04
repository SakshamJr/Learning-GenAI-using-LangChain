import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Template
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant. Use simple English."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

chat_history = []

# Load history
if os.path.exists("chat_history.txt"):
    with open("chat_history.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Human:"):
                chat_history.append(HumanMessage(content=line.replace("Human:", "").strip()))
            elif line.startswith("AI:"):
                chat_history.append(AIMessage(content=line.replace("AI:", "").strip()))

while True:
    query = input("You: ")
    if query == "exit":
        break

    prompt = chat_template.invoke({
        "chat_history": chat_history,
        "query": query
    })

    response = model.invoke(prompt)

    print("AI:", response.content)

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))

# Save history
with open("chat_history.txt", "w", encoding="utf-8") as f:
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            f.write(f"Human: {msg.content}\n")
        else:
            f.write(f"AI: {msg.content}\n")