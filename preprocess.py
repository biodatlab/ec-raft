import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
from client import Client
from prompt_gen.chain_of_thought_prompt_generator import ChainOfThoughtPromptGenerator
from prompt_gen.inference_prompt_generator import InferencePromptGenerator
import os
import argparse
import uuid
import json
import time
from vertexai.batch_prediction import BatchPredictionJob
import vertexai

PROJECT_ID = "<PROJECT_ID>" 
BUCKET_NAME = "<BUCKET_NAME>"
DATASET_NAME = "biodatlab/ec-raft-dataset"
OUTPUT_DIR = f"./data/no_exp_name"

def set_output_dir(experiment_name):
    """Set the global output directory for the experiment."""
    global OUTPUT_DIR
    OUTPUT_DIR = f"./data/{experiment_name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def initialize_components():
    """Initialize ChromaDB, models, and other components."""
    client = chromadb.PersistentClient(path="./clinical_trials_chroma_all")
    embed_model = SentenceTransformer("malteos/scincl")
    collection = client.get_or_create_collection("clinical_trials_studies")
    
    client_obj = Client(
        client=client,
        collection=collection,
        embed_model=embed_model
    )
    
    prompt_gen = ChainOfThoughtPromptGenerator(client=client_obj)
    dataset = load_dataset(DATASET_NAME)
    
    return client_obj, prompt_gen, dataset

def generate_dataset_with_top_n(n, split, experiment_name):
    """Generate dataset with top-N similar studies for the specified split."""
    _, prompt_gen, ravis_dataset = initialize_components()
    
    processed_data = []
    
    print(f"Processing {split} split with top-{n} similar studies...")
    for study in tqdm(ravis_dataset[split]):
        info_for_prompt = prompt_gen.extract_study_info(study, top_n=n)
        
        if info_for_prompt:
            related_studies_context, title, description, desired_criteria = info_for_prompt
            messages = prompt_gen.get_messages_for_CoT_huggingface(
                related_studies_context, title, description, desired_criteria
            )
            
            processed_data.append({
                'related_studies_context': related_studies_context,
                'title': title,
                'description': description,
                'desired_criteria': desired_criteria,
                'messages': messages
            })

    # Create DataFrame and save
    final_df = pd.DataFrame(processed_data)
    output_filename = os.path.join(OUTPUT_DIR, 'raw_data.pkl')
    # final_df.to_pickle(output_filename)
    
    print(f"Dataset generation completed. Saved {len(final_df)} records to {output_filename}")
    return final_df

def create_batch_prompts(df):
    """Create batch prompts for Gemini processing."""
    batch_prompts = []
    
    for _, row in df.iterrows():
        prompt = ChainOfThoughtPromptGenerator.user_prompt_template(
            row['related_studies_context'], 
            row['title'], 
            row['description'], 
            row['desired_criteria']
        )
        
        request_format = {
            "id": row["uuid"],
            "request": {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "system_instruction": {"parts": [{"text": ChainOfThoughtPromptGenerator.system_prompt}]},
            },
        }
        batch_prompts.append(request_format)
    
    return batch_prompts

def upload_prompts_to_bucket(batch_prompts, experiment_name):
    """Upload prompts to Google Cloud Storage bucket."""
    local_file = os.path.join(OUTPUT_DIR, "batch_prompts.jsonl")
    remote_path = f"gs://{BUCKET_NAME}/{experiment_name}_prompts.jsonl"
    
    with open(local_file, "w") as f:
        for prompt in batch_prompts:
            f.write(json.dumps(prompt) + "\n")
    
    print("Uploading prompts to bucket...")
    os.system(f"gsutil cp {local_file} {remote_path}")
    
    return remote_path

def process_batch_responses(batch_prediction_job):
    """Process and parse batch prediction responses."""
    output_location = batch_prediction_job.output_location + "/predictions.jsonl"
    local_output_path = os.path.join(OUTPUT_DIR, "batch_responses.jsonl")
    
    os.system(f"gsutil cp {output_location} {local_output_path}")
    
    responses_dict = {}
    with open(local_output_path, "r") as f:
        for line in f:
            try:
                response_data = json.loads(line)
                unique_id = response_data.get("id")
                response = response_data.get("response", {})
                
                if isinstance(response, dict) and "candidates" in response:
                    text_response = response["candidates"][0]["content"]["parts"][0]["text"]
                    responses_dict[unique_id] = text_response
                else:
                    print(f"No valid response found for ID {unique_id}")
                    
            except Exception as e:
                print(f"Error parsing response for ID {unique_id}: {e}")
    
    return responses_dict

def process_batch_with_gemini(df, experiment_name):
    """Process DataFrame with Gemini batch prediction."""
    if "uuid" not in df.columns:
        df["uuid"] = [str(uuid.uuid4()) for _ in range(len(df))]
        df.to_pickle(os.path.join(OUTPUT_DIR, "data_with_uuids.pkl"))
    
    if "response" not in df.columns:
        df["response"] = None

    df_to_process = df[df['response'].isna()]
    
    if len(df_to_process) == 0:
        print("All rows already processed!")
        return df
    
    print(f"Processing {len(df_to_process)} rows with Gemini...")
    
    batch_prompts = create_batch_prompts(df_to_process)
    
    if not batch_prompts:
        print("No prompts to process")
        return df
    
    print(f"Created {len(batch_prompts)} prompts")
    
    input_jsonl_path = upload_prompts_to_bucket(batch_prompts, experiment_name)
    
    print("Submitting batch prediction job...")
    batch_prediction_job = BatchPredictionJob.submit(
        source_model="gemini-1.5-flash-002",
        input_dataset=input_jsonl_path,
        output_uri_prefix=f"gs://{BUCKET_NAME}/output/",
    )

    print(f"Job resource name: {batch_prediction_job.resource_name}")
    print(f"Model resource name: {batch_prediction_job.model_name}")

    print("Waiting for job completion...")
    while not batch_prediction_job.has_ended:
        time.sleep(30)
        batch_prediction_job.refresh()
        print(f"Job status: {batch_prediction_job.state.name}")

    if batch_prediction_job.has_succeeded:
        print("Batch prediction completed successfully!")
        
        responses_dict = process_batch_responses(batch_prediction_job)
        
        if responses_dict:
            df["response"] = df["uuid"].map(responses_dict)
            print(f"Successfully mapped {len(responses_dict)} responses")
        
        return df
    else:
        print(f"Batch prediction job failed: {batch_prediction_job.error}")
        return None

def create_local_dataset(df, split='train'):
    """Process the responses and create a local parquet dataset."""
    print("Processing responses to create dataset...")
    
    if split == 'train':
        # For train split, use the original logic with response filtering and output mapping
        df_with_responses = df[df['response'].notna()]
        print(f"Found {len(df_with_responses)} rows with responses out of {len(df)} total rows")
        
        df_filtered = df_with_responses[
            df_with_responses['response'].str.contains('</STEP-BY-STEP-DERIVATION-FROM-TITLE-AND-DESCRIPTION>', na=False)
        ]
        print(f"Found {len(df_filtered)} rows with valid response format")
        
        if len(df_filtered) == 0:
            print("No valid responses found. Cannot create dataset.")
            return None
        
        print("Generating input and output columns...")
        df_filtered['input'] = df_filtered.apply(
            lambda x: InferencePromptGenerator.create_input(x['related_studies_context'], x['title'], x['description']), 
            axis=1
        )
        df_filtered['output'] = df_filtered.apply(
            lambda x: InferencePromptGenerator.format_output(x['response'], x['desired_criteria']), 
            axis=1
        )
        
        output_file = os.path.join(OUTPUT_DIR, "final_dataset.parquet")
        df_filtered.to_parquet(output_file, index=False)
        
        print(f"Dataset saved locally as '{output_file}' with {len(df_filtered)} examples")
        return df_filtered
    
    else:
        # For test and validation splits, only map input without filtering responses
        print(f"Processing {split} split - only generating input column...")
        
        print("Generating input column...")
        df['input'] = df.apply(
            lambda x: InferencePromptGenerator.create_input(x['related_studies_context'], x['title'], x['description']), 
            axis=1
        )
        
        output_file = os.path.join(OUTPUT_DIR, "final_dataset.parquet")
        df.to_parquet(output_file, index=False)
        
        print(f"Dataset saved locally as '{output_file}' with {len(df)} examples")
        return df

def main():
    """Main function to handle command line arguments and orchestrate the process."""
    parser = argparse.ArgumentParser(
        description='Generate dataset with top-N similar studies and process with Gemini'
    )
    parser.add_argument(
        '--n', type=int, default=5, 
        help='Number of top similar studies to include'
    )
    parser.add_argument(
        '--split', type=str, default='train', 
        choices=['train', 'test', 'validation'],
        help='Dataset split to process (train/test/validation)'
    )
    parser.add_argument(
        '--process', action='store_true', 
        help='Process the generated dataset with Gemini'
    )
    parser.add_argument(
        '--name', type=str, required=True,
        help='Experiment/dataset name (used for all files and final dataset)'
    )
    parser.add_argument(
        '--create_dataset', action='store_true', 
        help='Create local parquet dataset after processing'
    )
    
    args = parser.parse_args()
    
    experiment_name = f"{args.name}_top{args.n}_{args.split}"
    
    set_output_dir(experiment_name)
    
    print(f"Generating dataset for experiment: {experiment_name}")
    df = generate_dataset_with_top_n(args.n, args.split, experiment_name)

    if args.process and args.split == 'train':
        vertexai.init(project=PROJECT_ID, location="us-central1")
        print(f"Processing dataset with Gemini...")
        
        df = process_batch_with_gemini(df, experiment_name)
        
        if df is not None:
            output_file = os.path.join(OUTPUT_DIR, "data_with_responses.parquet")
            df.to_parquet(output_file)
            print(f"Processing completed. Results saved to {output_file}")
        else:
            print("Gemini processing failed. Cannot create dataset.")
    elif args.split != 'train':
        print("Skipping Gemini processing for test and validation splits")

    if args.create_dataset:
        dataset = create_local_dataset(df, args.split)
        if dataset is not None:
            print("Dataset creation completed successfully!")
        else:
            print("Dataset creation failed.")
    

if __name__ == "__main__":
    main()
