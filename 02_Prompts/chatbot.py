from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen3-Coder-Next", task="text-generation")
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(
        content="You are a helpful assistant that replies in short and concise answers."
    )
]

while True:
    user_input = input("You: ")
    messages.append(HumanMessage(content=user_input))
    if user_input == "exit":
        print("Thanks for using our ChatApp.")
        break
    result = model.invoke(messages)  # Its stateless so it doesnt retain context
    messages.append(AIMessage(content=result.content))  # Adding chat memory
    print(f"AI: {result.content}")
# print(f"Chat history: {chat_history}")
