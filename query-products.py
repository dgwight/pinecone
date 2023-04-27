# Query by user factors
import os
import time
import pinecone
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Load Pinecone API key
api_key = os.getenv('PINECONE_API_KEY')
# Set Pinecone environment.
env = os.getenv('PINECONE_ENVIRONMENT')

pinecone.init(api_key=api_key, environment=env)

index_name = 'shopping-cart-demo'

index = pinecone.Index(index_name=index_name)

start_time = time.process_time()
query_results = index.query(queries=user_factors[:-1].tolist(), top_k=10)
print("Time needed for retrieving recommended products using Pinecone: " + str(time.process_time() - start_time) + ' seconds.\n')

for _id, res in zip(user_ids, query_results.results):
    print(f'user_id={_id}')
    df = pd.DataFrame(d
        {
            'products': [match.id for match in res.matches],
            'scores': [match.score for match in res.matches]
        }
    )
    print("Recommendation: ")
    display(df)
    print("Top buys from the past: ")
    display(products_bought_by_user_in_the_past(_id, top=15))
