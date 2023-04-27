import os
from dotenv import load_dotenv

load_dotenv()

print(os.getenv('PINECONE_API_KEY'))
