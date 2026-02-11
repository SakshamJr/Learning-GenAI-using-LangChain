from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1", task="text-generation")

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name: str = Field(description="Name of the Person")
    age: int = Field(gt=18, description="Age of the Person")
    city: str = Field(description="Name of the City the Person belongs to.")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the Name, Age, and City of a Fictional {place} Person.\n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# prompt = template.invoke({"place": "Nepali"})

# # print(prompt)
# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser
final_result = chain.invoke({"place":"Nepal"})
print(final_result)
