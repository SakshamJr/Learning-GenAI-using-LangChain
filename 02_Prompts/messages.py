from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen3-Coder-Next", task="text-generation")
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(
        content="You are a helpful assistant that responses in short and concise manner."
    ),
    HumanMessage(content="Tell me about LangChain"),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)