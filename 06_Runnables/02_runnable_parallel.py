from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

# Model 1
llm1 = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation")
model_tweet = ChatHuggingFace(llm=llm1)

# Model 2
llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation"
)
model_linkedin = ChatHuggingFace(llm=llm2)

# Parser
parser = StrOutputParser()

# Prompt Template
prompt1 = PromptTemplate(
    template="Write a short text-only Tweet about {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Write a short, one paragraph, plain-text LinkedIn post about {topic}", input_variables=["topic"]
)

chain = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt1, model_tweet, parser),
        "linkedin": RunnableSequence(prompt2, model_linkedin, parser),
    }
)

result = chain.invoke({"topic": "Convolutional Neural Network"})
print(f"Tweet:\n{result['tweet']}")
print(f"\nLinkedIn Post:\n{result['linkedin']}")
