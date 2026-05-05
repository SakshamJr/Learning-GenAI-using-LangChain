# Allows to apply custom python functions within an AI pipeline
# Acts as a middleware between different AI components
# enabling preprocessing, transformation, API calls, filtering, 
# and post-processing in a LangChain workflow

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence, RunnablePassthrough

# def word_counter(text):
#     return len(text.split())

llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct', task = 'text-generation')
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template = "Write a short joke about {topic}",
    input_variables=['topic']
)

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'count':RunnableLambda(lambda x:len(x.split()))
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic":"Maths"})

print(f"Joke:\n{result['joke']}\n")
print(f"Word Count: {result['count']}")