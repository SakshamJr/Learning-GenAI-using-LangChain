from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen3-Coder-Next", task="text-generation")
model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful customer support assistant. Dont use Emojis and use plain understandable English."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
    ]
)

chat_history = []
with open("02_Prompts/chat_history.txt") as f:
    chat_history.extend(f.readlines())
if chat_history:
    print("Restored Previous Conversation.\n")
while True:
    query = str(input("You: "))
    if query == "exit":
        print("Thanks for using our ChatBot!")
        break
    chat_template.append(HumanMessage(content=query))
    prompt = chat_template.invoke({"chat_history": chat_history, "query": query})

    response = model.invoke(prompt)
    print(f"AI: {response.content}")
    chat_template.append(AIMessage(content=response.content))
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))

with open("02_Prompts/chat_history.txt", "a",encoding="utf-8") as f:
    # Simple text format
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            f.write(f"Human: {msg.content}\n")
        elif isinstance(msg, AIMessage):
            f.write(f"AI: {msg.content}\n")
