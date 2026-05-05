from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableBranch,
    RunnablePassthrough,
    RunnableLambda,
)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt_report = PromptTemplate(
    template="Generate a report on {topic}", input_variables=["topic"]
)

prompt_summarize = PromptTemplate(
    template="Summarize the text to below 200 words:\n{text}", input_variables=["text"]
)

gen_chain = RunnableSequence(prompt_report, model, parser)

branch_chain = RunnableBranch(
    (
        RunnableLambda(lambda x: len(x.split()) > 200),
        RunnableSequence(prompt_summarize, model, parser),
    ),
    RunnablePassthrough()
)

final_chain = RunnableSequence(gen_chain, branch_chain)

result = final_chain.invoke({"topic":"Artificial Intelligence"})

print(result)

# final_chain.get_graph().print_ascii()