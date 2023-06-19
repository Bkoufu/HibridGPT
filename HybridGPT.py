#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma

import numpy as np
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
from typing import Tuple, List
from langchain.docstore.document import Document
import  faiss 
import pickle  
import json
import logging
import sqlite3
import time
from publicGPT import get_response_from_openai

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))


from constants import CHROMA_SETTINGS

# class is implemented using Faiss for efficient similarity search and indexing of dense vectors.
class VectorStore:
    def __init__(self, index_path, ids_path):
        self.index_path = index_path
        self.ids_path = ids_path
        self.index = None
        self.ids = []
        
    def store_embedding(self, query, embedding):
        if query in self.ids:
            print("The embedding for the query already exists in the index.")
            return 

        if self.index is None:
            self.index = faiss.IndexFlatIP(embedding.shape[1])

        embedding = np.array(embedding).reshape(-1, embedding.shape[1]).astype('float32')
        self.index.add(embedding)
        self.ids.append(query)
        self.save_index()
        self.save_ids()


    def save_index(self):
        faiss.write_index(self.index, self.index_path)
    
    def save_ids(self):
        with open(self.ids_path, 'wb') as f:
            pickle.dump(self.ids, f)
        
    def find_k_similar_requests(self, query, k):
        # Check if query exists in the index
        try:
            query_index = self.ids.index(query)
        except ValueError:
            print("Query not found in the vector store.")
            return []
            
        # Check if the FAISS index can reconstruct vectors
        try:
            query_embedding = self.index.reconstruct(query_index)
        except RuntimeError:
            print("Cannot reconstruct vector from the FAISS index. \
                This might be due to the type of the index (like IndexFlatL2 or IndexIVFFlat),\
                where the full vector isn't stored in the index to save memory.")
            return []

        # Perform the search
        D, I = self.index.search(query_embedding.reshape(1, -1), k+1)

        # No need to convert distances to similarities. Use inner product scores directly.
        similar_requests = [{'id': self.ids[i], 'similarity': d} for d, i in zip(D[0], I[0]) if i != query_index]

        return similar_requests

    
    def load_index(self):
        self.index = faiss.read_index(self.index_path)

    def load_ids(self):
        with open(self.ids_path, 'rb') as f:
            self.ids = pickle.load(f)

    
 #class is implemented using SQLite for storing and retrieving the cached responses.   
class CacheStorage:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def initialize_database(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS cache (query TEXT PRIMARY KEY, answer TEXT, docs TEXT)"
        )
        self.conn.commit()

    def is_cached(self, query):
        self.cursor.execute("SELECT COUNT(*) FROM cache WHERE query=?", (query,))
        count = self.cursor.fetchone()[0]
        return count > 0

    def get_cached_response(self, query):
        self.cursor.execute("SELECT answer, docs FROM cache WHERE query=?", (query,))
        result = self.cursor.fetchone()
        if result is not None:
            answer,docs_json = result[0],result[1]
            docs = json.loads(docs_json)  # Deserialize the docs JSON string back into a list
            return answer, docs
        return None, None
    
    @staticmethod
    def document_to_dict(doc):
        return {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
    def cache_response(self, query, answer, docs):
        docs_json = json.dumps(docs, default=self.document_to_dict)  # 
        self.cursor.execute("INSERT INTO cache (query, answer, docs) VALUES (?, ?, ?)", (query, answer, docs_json))
        self.conn.commit()

    def save_cache(self):
        self.cursor.execute("SELECT query, answer, docs FROM cache")
        results = self.cursor.fetchall()
        cache = {}
        for row in results:
            query, answer, docs = row[0], row[1], row[2]
            cache[query] = (answer, docs)
        cache_data = pickle.dumps(cache)
        self.cursor.execute("UPDATE cache SET answer=?, docs=?", (sqlite3.Binary(cache_data),))
        self.conn.commit()

    def load_cache(self):
        self.cursor.execute("SELECT answer FROM cache")
        results = self.cursor.fetchall()
        if len(results) > 0:
            cache_data = results[0][0]
            cache = pickle.loads(cache_data)
            for query, (answer, docs) in cache.items():
                self.cache_response(query, answer, docs)

    def close_connection(self):
        self.cursor.close()
        self.conn.close()


# Prepare necessary models and services 
def prepare_services(hide_source: bool):
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, 
                embedding_function=embeddings, 
                client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    try:
        n_cpus=len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpus  = os.cpu_count()

    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_threads=n_cpus, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_threads=n_cpus, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    return qa,embeddings


# Initialize vector store
def init_vector_store(query, embeddings):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_dir, "vectorstore/faiss.index")
    ids_path = os.path.join(current_dir, "vectorstore/ids.pkl")
    vector_store = VectorStore(index_path=index_path, ids_path=ids_path) 
    embedding = embeddings.embed_query(query)
    embedding = np.array(embedding).reshape(-1, len(embedding)).astype('float32')
    vector_store.store_embedding(query, embedding)
    return vector_store

# Initialize cache storage
def init_cache_storage():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(current_dir, "Cache/cache.db")
    cache_storage = CacheStorage(db_path=relative_path)
    cache_storage.initialize_database()
    return cache_storage

# Get response from cache storage or compute it using qa model
def get_or_compute_response(cache_storage, query, qa, hide_source,use_openai):
    # When use_openai is set to True, bypass the cache and directly access OpenAI for an answer
    if use_openai:
        answer = get_response_from_openai(query)
        docs = []
    else:
        # Check if the response is already cached
        if cache_storage.is_cached(query):
            # Retrieve the cached response
            answer, docs = cache_storage.get_cached_response(query)
        else:
            # Compute the response using the chain
            res = qa(query)
            answer = res['result']
            docs = [] if hide_source else res['source_documents']
            # Cache the response
            cache_storage.cache_response(query, answer, docs)
            logging.info('response are: %s', answer, docs)
    return answer, docs

# Chat document function - composed of several smaller functions
def chatDocument(query: str, hide_source: bool, use_openai:bool) -> Tuple[str, List[Document]]:
    try:
        qa,embeddings= prepare_services(hide_source)
        vector_store = init_vector_store(query, embeddings)
        cache_storage = init_cache_storage()
        k_similar_requests = vector_store.find_k_similar_requests(query, k=3)
        start_time = time.time()
        answer, docs = get_or_compute_response(cache_storage, query, qa, hide_source,use_openai)
        elapsed_time = time.time() - start_time
        logging.info('Time taken: %.2f seconds', elapsed_time)

        if any(request['similarity'] > 0.9 for request in k_similar_requests):
            cache_storage.save_cache()
        cache_storage.close_connection()
        
        return answer, docs

    except Exception as e:
        error_message = f"Error occurred :{str(e)}"
        raise ValueError(error_message)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


#if __name__ == "__main__":
#    main()
