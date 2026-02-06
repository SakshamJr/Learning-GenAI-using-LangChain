from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-2",temperature=0.7)
response = model.invoke("What is the capital of Nepal?")
print(response.content)