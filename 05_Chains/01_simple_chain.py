# Imports
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prompt Template
prompt = PromptTemplate(
    template="Generate 5 short interesting facts about {topic}. The response should be plain-text.", input_variables=["topic"]
)

# LLM
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1", task="text-generation")
model = ChatHuggingFace(llm=llm)

# Output Parser
parser = StrOutputParser()

# Chain (LCEL)
chain = prompt | model | parser

# result = chain.invoke({"topic": "Nepal"})

# print(result)

# Visualize the chain
chain.get_graph().print_ascii()