from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4",temperature=0.7,max_tokens=10)

response = model.invoke("What is the capital of France?")
print(response)
# print(response.content)