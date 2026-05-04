from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Model Definition
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1", task="text-generation")
model = ChatHuggingFace(llm=llm)

# Template

prompt1 = PromptTemplate(
    template="Write a short report on {topic}", input_variables=["topic"]
)

# Template 2
prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text:\n{text}",
    input_variables=["text"],
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

# result = chain.invoke({'topic':'Tourism in Nepal'})

# print(result)
chain.get_graph().print_ascii()