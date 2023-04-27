import os
import pinecone
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# Load Pinecone API key
api_key = os.getenv('PINECONE_API_KEY')
# Set Pinecone environment.
env = os.getenv('PINECONE_ENVIRONMENT')

print(api_key, env)

# pinecone.init(api_key=api_key, environment=env)
#
# index_name = 'shopping-cart-demo'
#
# # Make sure service with the same name does not exist
# if index_name in pinecone.list_indexes():
#     pinecone.delete_index(index_name)
# pinecone.create_index(name=index_name, dimension=100)
#
# index = pinecone.Index(index_name=index_name)

def display(data):
    pd.DataFrame(data)

display({'col1': [1, 2], 'col2': [3, 4]})



