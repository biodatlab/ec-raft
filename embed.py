import chromadb
from datasets import load_dataset
import json
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./clinical_trials_chroma_all")
model = SentenceTransformer("malteos/scincl")
collection_studies = client.get_or_create_collection("clinical_trials_studies")
ravis_dataset = load_dataset("swissnp/ec-raft-clinicaltrials-gov")

def embed_studies_from_dataset(dataset, batch_size=32):
    batch_texts = []       
    batch_metadata = []    
    batch_documents = []   
    batch_ids = []         
    index = 1
    length = len(dataset['train'])
    
    for study in dataset['train']:
            metadata = json.loads(study['metadata'])
            try:
                title = metadata.get('Official_title', '') or metadata.get('Brief_Title', '')
            except:
                print(metadata)
                continue
            detailed_description = study.get('data', '')

            if not title or not detailed_description:
                continue
            
            concatenated_text = f"{title} [SEP] {detailed_description}"
            
            batch_texts.append(concatenated_text)
            batch_metadata.append({
                "nctId": metadata.get("NCT_ID", "unknown"),
                "officialTitle": title,
                "detailedDescription": detailed_description,
                "jsonMetadata": json.dumps(metadata, ensure_ascii=True)
            })
            batch_documents.append(json.dumps({
                "metadata": metadata,
                "description": study.get('data', ''),
                "criteria": study.get('criteria', '')
                },ensure_ascii=True))
            batch_ids.append(metadata.get("NCT_ID", "unknown"))

            if len(batch_texts) == batch_size:
                process_batch(batch_texts, batch_documents, batch_ids, batch_metadata)
                print(f"Processed {len(batch_texts)} studies. {index}/{length}")
                batch_texts.clear()
                batch_documents.clear()
                batch_metadata.clear()
                batch_ids.clear()
            index += 1

    if batch_texts:
        process_batch(batch_texts, batch_documents, batch_ids, batch_metadata)

def process_batch(texts, documents, ids, metadatas):
    study_embeddings = model.encode(texts, batch_size=len(texts))
    collection_studies.add(
        embeddings=study_embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Processed and added batch of {len(texts)} studies to collection.")
embed_studies_from_dataset(ravis_dataset, batch_size=750)
print("Embedding and storing complete!")