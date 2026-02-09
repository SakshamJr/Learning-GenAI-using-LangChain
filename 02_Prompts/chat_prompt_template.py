from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen3-Coder-Next", task="text-generation")
model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful {domain} expert. Answer in short and concise way.",
        ),
        ("human", "Explain in simple terms what is {topic}"),
    ]
)

prompt = chat_template.invoke({"domain": "Physics", "topic": "Projectile Motion"})

print(prompt)
