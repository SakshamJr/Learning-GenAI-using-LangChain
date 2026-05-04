from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Objective: Generate Notes and Quiz both from a topic in parallel.

# Model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Output Parser
parser = StrOutputParser()

# Prompt Template: Generate Notes
prompt1 = PromptTemplate(
    template="What is {topic}? Give me a Short Note about it.",
    input_variables=["topic"],
)

# Prompt Template 2: Generate 4 Quiz Questions
prompt2 = PromptTemplate(
    template="Generate 4 Quiz Questions related to {topic}. The questions and answers should be short.",
    input_variables=["topic"],
)

# Prompt Template 3: Merge the Provided Notes and Quiz into a single document.\n notes -> {notes} and quiz - > {quiz}
prompt3 = PromptTemplate(
    template="Merge the Provided Notes and Quiz into a single document.\n notes -> {notes} and quiz - > {quiz}",
    input_variables=["notes", "quiz"],
)

parallel_chain = RunnableParallel(
    {"notes": prompt1 | model | parser, "quiz": prompt2 | model | parser}
)

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

# result = chain.invoke({"topic": "Nepal"})

# print(result)

chain.get_graph().print_ascii()
