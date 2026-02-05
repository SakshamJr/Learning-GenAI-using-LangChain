from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(repo_id= 'sentence-transformers/all-MiniLM-L6-v2')

text = "Saksham Sapkota is the Don of IOE."

result = embedding.embed_query(text=text)

print(str(result))