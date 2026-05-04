from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen3-Coder-Next", task="text-generation")
model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful AI Assistant that responds in short and concise answers. No emoji, no unnecessary formatting, just plain text.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
    ]
)

chat_history = []

# Load Chat History if any
if os.path.exists("02_Prompts/chat_history.txt"):
    print("Chat History Restored!")
    with open("02_Prompts/chat_history.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("User: "):
                chat_history.append(
                    HumanMessage(content=line.replace("User: ", "").strip())
                )
            elif line.startswith("AI: "):
                chat_history.append(AIMessage(content=line.replace("AI: ", "").strip()))

# Loop for Chatbot
while True:
    query = str(input("SakshamJr: "))
    if query == "exit":
        print("Thank You for using our ChatBot!\n")
        break

    prompt = chat_template.invoke({"chat_history": chat_history, "query": query})
    response = model.invoke(prompt)

    print(f"AI: {response.content}")

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))

# Save chat history in files
with open("02_Prompts/chat_history.txt", "w", encoding="utf-8") as f:
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            f.write(f"User: {msg.content}\n")
        elif isinstance(msg, AIMessage):
            f.write(f"AI: {msg.content}\n")
