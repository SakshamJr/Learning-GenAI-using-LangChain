from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name= 'sentence-transformers/all-MiniLM-L6-v2')
text = "Saksham Sapkota is the Don of IOE."
result = embedding.embed_query(text=text)
print(str(result))