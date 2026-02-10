from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict, Annotated, Optional, Literal

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", task="text-generation")
model = ChatHuggingFace(llm=llm)


class Review(TypedDict):
    key_themes: Annotated[
        list[str], "Write down all the key themes discussed in the review in a list"
    ]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[
        Literal["positive", "negative", "neutral"],
        "Return the sentiment of the review either positive, negative or neutral, nothing other than these three options.",
    ]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]


structured_model = model.with_structured_output(Review)
result = structured_model.invoke(
    """Very handy to use. It feels good in the hand. Very effective, much better than already known products. It is great that you can charge the device and does not require any batteries. Well made, saves space.

    very useful and handy

    the electric wires were peeking out and also the motor is not strong

    Received a defective product which is not working at all...bad experience

    It's nice but I hope its motor was stronger"""
)

# print(result["summary"])
# print(result["sentiment"])
print(result)
