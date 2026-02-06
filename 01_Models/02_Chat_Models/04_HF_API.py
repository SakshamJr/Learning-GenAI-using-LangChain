from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next",
    task="text-generation")

model = ChatHuggingFace(llm=llm)
result = model.invoke("Who is the prime minister of Nepal?")
print(result.content)