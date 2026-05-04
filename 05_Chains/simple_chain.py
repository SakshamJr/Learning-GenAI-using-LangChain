from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1", task="text-generation")
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}", input_variables=["topic"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "Nepal"})

print(result)

# chain.get_graph().print_ascii()