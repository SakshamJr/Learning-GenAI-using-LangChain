from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

# Model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Prompt Template
prompt1 = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)

# Prompt Template 2
prompt2 = PromptTemplate(
    template="Explain the following joke:\n{text}", input_variables=["text"]
)

# Parser
parser = StrOutputParser()

# Chain
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({"topic": "AI"})

print(result)
