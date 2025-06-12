import chromadb
from sentence_transformers import SentenceTransformer

class Client:
  def __init__(self, client: chromadb.Client, embed_model: SentenceTransformer, collection: str):
        self.client = client
        self.embed_model = embed_model
        self.collection = collection
        
  def retrieve_relevant_studies(self, title: str, description: str, existing_study: str, n_results=5):
      query = f'{title} [SEP] {description}'
      query_embedding = self.embed_model.encode(query).tolist()
      
      results = self.collection.query(
          query_embeddings=[query_embedding],
          n_results=n_results + 1,
      )
      
      filtered_results = []
      for id, distance, document in zip(results['ids'][0], results['distances'][0], results['documents'][0]):
          if id != existing_study:
              filtered_results.append({
                  "id": id,
                  "distance": distance,
                  "document": document,
              })
          
          if len(filtered_results) == n_results:
              break

      return filtered_results
  