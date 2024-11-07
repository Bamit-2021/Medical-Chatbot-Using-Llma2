from src.helper import load_pdf, text_split, download_hugging_face_embeddings
# from langchain.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
import pinecone
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)


# pc = Pinecone(api_key=PINECONE_API_KEY)

# index_name = 'medical-chatbot'

# pc.create_index(
#     name=index_name,
#     dimension=384,
#     metric='cosine',
#     spec=ServerlessSpec(
#         cloud='aws', 
#         region='us-east-1')
# )


extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="medical-chatbot"
#Creating Embeddings for Each of The Text Chunks & storing
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
