import os
import time
import random
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import itertools

order_products_train = pd.read_csv('data/order_products__train.csv')
order_products_prior = pd.read_csv('data/order_products__prior.csv')
products = pd.read_csv('data/products.csv')
orders = pd.read_csv('data/orders.csv')

order_products = order_products_train._append(order_products_prior)

customer_order_products = pd.merge(orders, order_products, how='inner',on='order_id')

# creating a table with "confidences"
data = customer_order_products.groupby(['user_id', 'product_id'])[['order_id']].count().reset_index()
data.columns=["user_id", "product_id", "total_orders"]
data.product_id = data.product_id.astype('int64')

# Create a lookup frame so we can get the product names back in readable form later.
products_lookup = products[['product_id', 'product_name']].drop_duplicates()
products_lookup['product_id'] = products_lookup.product_id.astype('int64')

data_new = pd.DataFrame([[data.user_id.max() + 1, 22802, 97],
                         [data.user_id.max() + 2, 26834, 89],
                         [data.user_id.max() + 2, 12590, 77]
                        ], columns=['user_id', 'product_id', 'total_orders'])

users = list(np.sort(data.user_id.unique()))
items = list(np.sort(products.product_id.unique()))
purchases = list(data.total_orders)

# create zero-based index position <-> user/item ID mappings
index_to_user = pd.Series(users)

# create reverse mappings from user/item ID to index positions
user_to_index = pd.Series(data=index_to_user.index + 1, index=index_to_user.values)

# create zero-based index position <-> item/user ID mappings
index_to_item = pd.Series(items)

# create reverse mapping from item/user ID to index positions
item_to_index = pd.Series(data=index_to_item.index, index=index_to_item.values)

# Get the rows and columns for our new matrix
products_rows = data.product_id.astype(int)
users_cols = data.user_id.astype(int)

# Create a sparse matrix for our users and products containing number of purchases
sparse_product_user = sparse.csr_matrix((purchases, (products_rows, users_cols)), shape=(len(items) + 1, len(users) + 1))
sparse_product_user.data = np.nan_to_num(sparse_product_user.data, copy=False)

sparse_user_product = sparse.csr_matrix((purchases, (users_cols, products_rows)), shape=(len(users) + 1, len(items) + 1))
sparse_user_product.data = np.nan_to_num(sparse_user_product.data, copy=False)



import implicit
from implicit import evaluation

#split data into train and test sets
train_set, test_set = evaluation.train_test_split(sparse_product_user, train_percentage=0.9)

# initialize a model
model = implicit.als.AlternatingLeastSquares(factors=100,
                                             regularization=0.05,
                                             iterations=50,
                                             num_threads=1)

alpha_val = 15
train_set = (train_set * alpha_val).astype('double')

# train the model on a sparse matrix of item/user/confidence weights
model.fit(train_set, show_progress = True)

# test_set = (test_set * alpha_val).astype('double')
# evaluation.ranking_metrics_at_k(model, train_set.T, test_set.T, K=100,
#                          show_progress=True, num_threads=1)

print(model.item_factors[1:3])





import pinecone
from dotenv import load_dotenv

load_dotenv()

# Load Pinecone API key
api_key = os.getenv('PINECONE_API_KEY')
# Set Pinecone environment.
env = os.getenv('PINECONE_ENVIRONMENT')

pinecone.init(api_key=api_key, environment=env)

index_name = 'shopping-cart-demo'

# Make sure service with the same name does not exist
if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)
pinecone.create_index(name=index_name, dimension=100)

index = pinecone.Index(index_name=index_name)



# Get all of the items
all_items = [title for title in products_lookup['product_name']]

# Transform items into factors
items_factors = model.item_factors

def display(data):
    print(data)

# Prepare item factors for upload
items_to_insert = list(zip(all_items, items_factors[1:].tolist()))
display(items_to_insert[:2])

def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


# print('Index statistics before upsert:', index.describe_index_stats())

for e, batch in enumerate(chunks([(ii[:64],x) for ii,x in items_to_insert])):
    index.upsert(vectors=batch)

print('Index statistics after upsert:', index.describe_index_stats())

def products_bought_by_user_in_the_past(user_id: int, top: int = 10):

    selected = data[data.user_id == user_id].sort_values(by=['total_orders'], ascending=False)

    selected['product_name'] = selected['product_id'].map(products_lookup.set_index('product_id')['product_name'])
    selected = selected[['product_id', 'product_name', 'total_orders']].reset_index(drop=True)
    if selected.shape[0] < top:
        return selected

    return selected[:top]

data.tail()
