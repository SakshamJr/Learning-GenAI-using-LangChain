from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about Virat Kohli"

doc_embeddings = embedding.embed_documents(documents)
query_embeddings = embedding.embed_query(query)

# print(np.array(doc_embeddings).shape)
# print(np.array(query_embeddings).shape)

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(f"Query: {query}")
print(f"Similar Document: {documents[index]}")
print(f"Similarity Score: {score}")