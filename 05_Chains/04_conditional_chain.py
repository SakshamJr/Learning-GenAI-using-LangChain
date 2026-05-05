# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
# from langchain_core.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# from typing import Literal

# # # Review Sentiment Output Structure using TypedDict
# # class Review(TypedDict):
# #     sentiment: Annotated[
# #         Literal["positive", "negative"],
# #         "Return the sentiment of the review either positive or negative, no other text than these two options",
# #     ]


# class Review(BaseModel):
#     sentiment: Literal["positive", "negative"] = Field(
#         description="Give the sentiment of the feedback."
#     )


# parser2 = PydanticOutputParser(pydantic_object=Review)

# # Model
# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation"
# )
# model = ChatHuggingFace(llm=llm)


# # Parser
# parser = StrOutputParser()

# # Prompt Template
# prompt1 = PromptTemplate(
#     template="""Classify the sentiment of this review:
#     {review}

#     Return output strictly in this JSON format:
#     {{"sentiment": "positive"}} OR {{"sentiment": "negative"}}
    
#     Do NOT return a list.
#     {format_instruction}
#     """,
#     input_variables=["review"],
#     partial_variables={"format_instruction": parser2.get_format_instructions()},
# )

# # Positive Review
# prompt2 = PromptTemplate(
#     template="Write a short paragraph of suitable positive response to this review:\n{review}\nNote: You should sound helpful, responsbile, happy, and humble.",
#     input_variables=["review"],
# )

# # Negative Review
# prompt3 = PromptTemplate(
#     template="Write a short paragraph of suitable apologetic response to this review:\n{review}\nNote: You should sound responsbile and curious about their problem.",
#     input_variables=["review"],
# )

# classifier_chain = prompt1 | model | parser2

# branch_chain = RunnableBranch(
#     (lambda x: x.sentiment == "positive", prompt2 | model | parser),
#     (lambda x: x.sentiment == "negative", prompt3 | model | parser),
#     RunnableLambda(lambda x: "Could not find sentiment."),
# )

# chain = classifier_chain | branch_chain

# result = chain.invoke({"review": "This is a terrible phone, the battery drains fast."})
# print(result)

# # chain.get_graph().print_ascii()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

# ---------------------------
# Schema
# ---------------------------
class Review(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback."
    )

parser2 = PydanticOutputParser(pydantic_object=Review)
parser = StrOutputParser()

# ---------------------------
# Model
# ---------------------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# ---------------------------
# Prompts
# ---------------------------
prompt1 = PromptTemplate(
    template="""Classify the sentiment of this review:
{review}

Return output strictly in this JSON format:
{{"sentiment": "positive"}} OR {{"sentiment": "negative"}}

Do NOT return a list.
{format_instruction}
""",
    input_variables=["review"],
    partial_variables={"format_instruction": parser2.get_format_instructions()},
)

prompt2 = PromptTemplate(
    template="""Write a short paragraph of a suitable positive response to this review:
{review}

Sound helpful, responsible, happy, and humble.""",
    input_variables=["review"],
)

prompt3 = PromptTemplate(
    template="""Write a short paragraph of a suitable apologetic response to this review:
{review}

Sound responsible and curious about their problem.""",
    input_variables=["review"],
)

# ---------------------------
# Chains
# ---------------------------

# Step 1: Sentiment classifier
classifier_chain = prompt1 | model | parser2

# Step 2: Response generators (ensure correct input shape)
positive_chain = (
    RunnableLambda(lambda x: {"review": x["review"]})
    | prompt2 | model | parser
)

negative_chain = (
    RunnableLambda(lambda x: {"review": x["review"]})
    | prompt3 | model | parser
)

# Step 3: Branching logic
branch_chain = RunnableBranch(
    (lambda x: x["sentiment"].sentiment == "positive", positive_chain),
    (lambda x: x["sentiment"].sentiment == "negative", negative_chain),
    RunnableLambda(lambda x: "Could not determine sentiment."),
)

# Step 4: Combine original review + sentiment
chain = RunnableParallel(
    sentiment=classifier_chain,
    review=lambda x: x["review"]
) | branch_chain

# ---------------------------
# Run
# ---------------------------
review_input = "This is an amazing phone, the battery is solid."
result = chain.invoke({"review": review_input})

print(result)