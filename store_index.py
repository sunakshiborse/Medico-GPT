from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)


index_name="medico-gpt"


# Check if the index exists
if index_name not in pc.list_indexes().names():
    dimension = 384
    metric = "cosine"
    pod_type = "starter"
    # Index already exists, continue with storing embeddings
    docsearch = pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)