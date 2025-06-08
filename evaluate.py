import json
import pandas as pd
import time
from vertexai.batch_prediction import BatchPredictionJob
import os
import uuid
import vertexai
import re
import numpy as np
from bert_score import score
from json_repair import repair_json
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel
from typing import List
from tqdm import tqdm
import argparse

PROJECT_ID = "<PROJECT_ID>" 
BUCKET_NAME = "<BUCKET_NAME>"  

class criteria_item(BaseModel):
    reference: str
    predicted: List[str]
    reason: str
    match_score: int

class sex(BaseModel):
    reference: str
    predicted: List[str]
    reason: str
    match_score: int

class age(BaseModel):
    reference: str
    predicted: List[str]
    reason: str
    match_score: int

class accept_healthy_volunteer(BaseModel):
    reference: str
    predicted: List[str]
    reason: str
    match_score: int

class unmatched_predicted_criteria(BaseModel):
    unmatched_predicted_inclusion_criteria: List[str]
    unmatched_predicted_exclusion_criteria: List[str]

class eligibility_criteria(BaseModel):
    inclusion_criteria: List[criteria_item]
    exclusion_criteria: List[criteria_item]
    sex: sex
    age: age
    accept_healthy_volunteer: accept_healthy_volunteer
    unmatched_predicted_criteria: unmatched_predicted_criteria

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

def construct_prompt(reference, predicted):
    return f"""<REFERENCE>
    {reference}
    </REFERENCE>

    ———

    <PREDICTED>
    {predicted}
    </PREDICTED>"""

def process_batch_free_text(df, reference_row, predicted_row, exp, max_retries=3):
    """Process batch free text evaluation using Gemini."""
    # Create experiment directory
    exp_dir = f"./results/{exp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    if "uuid" not in df.columns:
        df["uuid"] = [str(uuid.uuid4()) for _ in range(len(df))]
        df.to_parquet(f"{exp_dir}/{exp}_data_with_uuids.parquet")
    
    if "free_text_eval" not in df.columns:
        df["free_text_eval"] = None

    df_to_process = df[df['free_text_eval'].isna()]
    
    if len(df_to_process) == 0:
        print("All rows processed successfully!")
        return df

    system_prompt = """Please evaluate the clinical relevance of the following two eligibility criteria on a 4-point scale (0–3). Below is an example of a clinical situation by clinical relevance score and the corresponding EC pair.

- **Clinical relevance 3**: The two eligibility criteria are essentially identical clinically.
*Examples*:
    - EC1: "[exclusion] serum albumin is 2.4 g/dL or less"
    EC2: "[inclusion] serum albumin is 2.4 g/dL or more"
    - EC1: "Minimum Age : 18 Years"
    EC2: "Minimum Age : 18 Years"
- **Clinical relevance 2**: The two eligibility criteria have strong relevance due to factors such as disease progression, or epidemiology.
*Example*:
    - EC1: "[inclusion] 1 focal lesions on MRI (magnetic resonance imaging) studies; Each focal lesion must be 5 mm or more in size"
    EC2: "[exclusion] kellgren and Lawrence grade ≥ 3"
- **Clinical relevance 1**: The two eligibility criteria are not directly related, but still have some relevance due to factors such as general treatment plan, disease progression, or epidemiology.
*Example*:
    - EC1: "[inclusion] no concurrent major surgery"
    EC2: "[inclusion] histologically confirmed transitional cell carcinoma (TCC) of the urothelium"
- **Clinical relevance 0**: The eligibility criteria are irrelevant from a clinical perspective.
*Examples*:
    - EC1: "[exclusion] history of a severe allergic reaction with generalized urticaria, angioedema, or anaphylaxis in the 2 years prior to enrollment"
    EC2: "[inclusion] male condoms with spermicide"
    - EC1: "Minimum Age : 18 Years"
    EC2: "Minimum Age : 65 Years"

Evaluation Process

For each reference criterion, compare it to the relevant predicted criterions. If no relevant predicted criterion exists, state this explicitly. The evaluation process is as follows:

- Recite the reference exact criterion, state explicitly if it is from [inclusion] or [exclusion].
- Search the predicted criteria list to identify the relevant matchs, regardless of order (comma separated), and explicitly state which part of the predicted criteria each match comes from ([inclusion], [exclusion], [age], [sex], [accepts healthy volunteers]).
- After that Recite the reference ##Sex, ##Ages, and ##Accepts Healthy Volunteers one at a time and compare with the relevant predicted ##Sex, ##Ages, ##Accepts Healthy Volunteers or Inclusion/Exclusion (comma separated).
- Provide a reason explaining how the criteria match or differ.
- Assign a match score (0–3) based on the clinical relevance of the predicted criterion to the reference criterion.
- If no predicted criterion matches the reference, state that explicitly and assign a score of 0.

At the end of the evaluation, please provide:

- Unmatched Predicted Criteria:
    - Unmatched Predicted Inclusion Criteria: List all predicted inclusion criteria that were not matched to any reference criteria (relevance score = 0). No explanation is needed—just list them (comma separated).
    - Unmatched Predicted Exclusion Criteria: List all predicted exclusion criteria that were not matched to any reference criteria (relevance score = 0). No explanation is needed—just list them (comma separated)."""

    batch_prompts = []
    for _, row in df_to_process.iterrows():
        reference = row[reference_row]
        predicted = row[predicted_row]
        criteria_pattern = re.compile(r'<FORMATTED_CRITERIA>(.*?)</FORMATTED_CRITERIA>', re.DOTALL)
        
        if isinstance(predicted, float):
            print(f"Predicted is float: {predicted}")
            continue
            
        criteria_match = criteria_pattern.search(str(predicted))
        if criteria_match:
            criteria_text = criteria_match.group(1).strip()
            criteria = construct_prompt(reference, criteria_text)
            request_format = {
                "id": row["uuid"],
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": criteria}]}],
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                },
            }
            batch_prompts.append(request_format)
        else:
            print('no criteria found')
            continue
            
    if len(batch_prompts) == 0:
        print("No prompts to process")
        return df

    # Upload to bucket and process
    local_prompts_file = f"{exp_dir}/{exp}_eval_prompts.jsonl"
    if os.path.exists(local_prompts_file):
        os.remove(local_prompts_file)
    
    input_jsonl_path = f"gs://{BUCKET_NAME}/{exp}_eval_prompts.jsonl"
    with open(local_prompts_file, "w") as f:
        for prompt in batch_prompts:
            f.write(json.dumps(prompt) + "\n")
    
    os.system(f"gsutil cp {local_prompts_file} {input_jsonl_path}")

    batch_prediction_job = BatchPredictionJob.submit(
        source_model="gemini-1.5-flash-002",
        input_dataset=input_jsonl_path,
        output_uri_prefix=f"gs://{BUCKET_NAME}/{exp}_eval_output/",
    )

    print(f"Free text job resource name: {batch_prediction_job.resource_name}")

    while not batch_prediction_job.has_ended:
        time.sleep(30)
        batch_prediction_job.refresh()
        print(f"Free text job status: {batch_prediction_job.state.name}")

    if batch_prediction_job.has_succeeded:
        print("Free text generation completed successfully!")
        output_location = batch_prediction_job.output_location + "/predictions.jsonl"
        local_output_path = f"{exp_dir}/{exp}_eval_responses.jsonl"

        os.system(f"gsutil cp {output_location} {local_output_path}")

        responses_dict = {}
        with open(local_output_path, "r") as f:
            for line in f:
                response_data = json.loads(line)
                unique_id = response_data.get("id")
                response = response_data.get("response", {})
                if isinstance(response, dict) and "candidates" in response:
                    try:
                        responses_dict[unique_id] = response["candidates"][0]["content"]["parts"][0]["text"]
                    except Exception as e:
                        print(f"Could not parse response for ID {unique_id}: {e}")

        if responses_dict:
            df["free_text_eval"] = df["uuid"].map(responses_dict)
        
        return df
    else:
        print(f"Free text job failed: {batch_prediction_job.error}")
        return None

def get_messages(user_input: str):
    """Get messages for JSON parsing."""
    example_json = """
{
  "inclusion_criteria": [
    {
      "reference": "given reference criteria",
      "predicted": ["given matching criteria", "given matching criteria"] or [""] (if no match),
      "reason": "given reason",
      "match_score": 3
    },{
      "reference": "given reference criteria",
      "predicted": [""] (if no match),
      "reason": "given reason",
      "match_score": 0
    },
   … more as given …
  ],
  "exclusion_criteria": [
   {
      "reference": "given reference criteria",
      "predicted": ["given matching criteria"] or [""] (if no match),
      "reason": "given reason",
      "match_score": 2
    },
    {
      "reference": "given reference criteria",
      "predicted": [""] (if no match),
      "reason": "given reason",
      "match_score": 0
    },
   … more as given …
  ],
  "sex": {
    "reference": "given reference",
    "predicted": ["given predicted"] or [""] (if no match),
    "reason": "given reason",
    "match_score": 0
  },
  "age": {
    "reference": "given reference",
    "predicted": ["given predicted"] or [""] (if no match),
    "reason": "given reason",
    "match_score": 2
  },
  "accept_healthy_volunteer": {
    "reference": "given reference",
    "predicted": ["given predicted"] or [""] (if no match),
    "reason": "given reason",
    "match_score": 1
  },
  "unmatched_predicted_criteria": {
    "unmatched_predicted_inclusion_criteria": ["given predicted inclusion unmatched", "given predicted inclusion unmatched"] or [""] (if no match),
    "unmatched_predicted_exclusion_criteria": ["given predicted exclusion unmatched"] or [""] (if no match)
  }
}
"""
    return [
        {
            "role": "system",
            "content": f"""Please parse the following given text below into the json: 
Example json: {example_json}
No need to come up with any other text. Just parse the given text into the json.
Make sure to parse every match pair in the given text into the json even if there are no matches in predicted."""
        },
        {
            "role": "user",
            "content": user_input + """
Don't need to come up with any other text. Just parse the given text into the json.
Make sure to parse every match pair in the given text into the json even if there are no matches in predicted."""
        },
    ]

def process_json_parsing_with_vllm(df, exp, model_path):
    """Process JSON parsing using vLLM."""
    json_schema = eligibility_criteria.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(
        json_schema,
        whitespace_pattern=r"[ ]?"
    )
    
    llm = LLM(
        model=model_path,
        max_model_len=20000,
        tensor_parallel_size=1
    )
    
    df = df[df['free_text_eval'].apply(lambda x: isinstance(x, str))]
    if 'retry_count' not in df.columns:
        df['retry_count'] = 0
    if 'response_json' not in df.columns:
        df['response_json'] = None

    batch_size = 1000
    max_retries = 3

    def process_batch_with_retries(batch_indices, is_retry=False):
        sampling_params = SamplingParams(
            max_tokens=8192,
            min_p=0.03,
            temperature=0.3 if not is_retry else 0.4,
            guided_decoding=guided_decoding_params,
        )

        tokenizer = llm.get_tokenizer()
        prompts = [tokenizer.apply_chat_template(get_messages(df.loc[idx, 'free_text_eval']), 
                                                add_generation_prompt=True, tokenize=False) 
                  for idx in batch_indices]
        outputs = llm.generate(prompts, sampling_params)
        
        to_retry = []
        for idx, output in zip(batch_indices, outputs):
            response = output.outputs[0].text
            
            try:
                json_response = json.loads(response)
                parsed_response = eligibility_criteria.model_validate(json_response)
                df.at[idx, 'response_json'] = json.dumps(parsed_response.model_dump())
            except Exception as e:
                if df.loc[idx, 'retry_count'] < max_retries:
                    df.loc[idx, 'retry_count'] += 1
                    to_retry.append(idx)
                else:
                    df.loc[idx, 'response_json'] = response
        
        return to_retry

    pending_retries = []
    with tqdm(total=len(df), desc="Processing JSON") as pbar:
        for i in range(0, len(df), batch_size):
            if len(pending_retries) >= batch_size:
                retry_batch = pending_retries[:batch_size]
                pending_retries = pending_retries[batch_size:]
                new_retries = process_batch_with_retries(retry_batch, is_retry=True)
                pending_retries.extend(new_retries)

            current_batch = df.index[i:i + batch_size].tolist()
            new_retries = process_batch_with_retries(current_batch)
            pending_retries.extend(new_retries)
            
            pbar.update(batch_size)

        while pending_retries:
            retry_batch = pending_retries[:batch_size]
            pending_retries = pending_retries[batch_size:]
            new_retries = process_batch_with_retries(retry_batch, is_retry=True)
            pending_retries.extend(new_retries)

    return df

def safe_json_load(json_string):
    """Safely load and repair JSON."""
    try:
        if not json_string or json_string.isspace():
            return {}
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError, AttributeError):
        try:
            repaired_string = repair_json(json_string)
            return json.loads(repaired_string)
        except (json.JSONDecodeError, TypeError, AttributeError):
            return {}

def safe_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0

def should_exclude(criterion):
    if not isinstance(criterion, str):
        return True
    criterion = criterion.strip()
    words = criterion.split()
    if len(criterion) <= 1:
        return True
    if len(words) == 1 and words[0].lower() == "none":
        return True
    if len(words) > 1 and words[0].lower() == "no":
        if any(kw in criterion.lower() for kw in ['match', 'relevant', 'criteria']):
            return True
    return False

def parse_response(data):
    """Flattens the JSON structure into a list of rows."""
    rows = []
    for key, value in data.items():
        if key in ["number_of_criteria", "unmatched_predicted_criteria"]:
            continue
        if isinstance(value, list):
            for item in value:
                try:
                    rows.append({
                        "category": key,
                        "match_score": safe_int(item.get("match_score")),
                        "predicted": item.get("predicted"),
                        "reason": item.get("reason"),
                        "reference": item.get("reference")
                    })
                except Exception as e:
                    print(f"Error parsing item: {e}")
        elif isinstance(value, dict):
            rows.append({
                "category": key,
                "match_score": safe_int(value.get("match_score")),
                "predicted": value.get("predicted"),
                "reason": value.get("reason"),
                "reference": value.get("reference")
            })
    return rows

def parse_json_from_df(df, json_column):
    rows = []
    for json_string in df[json_column]:
        data = safe_json_load(json_string)
        rows.extend(parse_response(data))
    return pd.DataFrame(rows)

def calculate_clean_unmatched_criteria(df):
    rows = []
    for _, row in df.iterrows():
        data = safe_json_load(row["response_json"])
        if "unmatched_predicted_criteria" in data:
            unmatched = data["unmatched_predicted_criteria"]
            exc = unmatched.get("unmatched_predicted_exclusion_criteria", [])
            inc = unmatched.get("unmatched_predicted_inclusion_criteria", [])
            clean_exc = [c for c in exc if not should_exclude(c)]
            clean_inc = [c for c in inc if not should_exclude(c)]
            rows.append({
                "clean_unmatched_exclusion_count": len(clean_exc),
                "clean_unmatched_inclusion_count": len(clean_inc),
                "total_unmatched_exclusion_count": len(exc),
                "total_unmatched_inclusion_count": len(inc),
                "excluded_exclusion_count": len(exc) - len(clean_exc),
                "excluded_inclusion_count": len(inc) - len(clean_inc)
            })
    return pd.DataFrame(rows)

def compute_precision_recall_direct(df, json_column="response_json"):
    """Compute precision and recall directly from the JSON responses."""
    criteria_keys = [
        "inclusion_criteria",
        "exclusion_criteria", 
        "sex",
        "age",
        "accept_healthy_volunteer"
    ]
    
    total_matches = 0
    total_reference = 0

    for _, row in df.iterrows():
        data = safe_json_load(row[json_column])
        if not data:
            continue

        for key in criteria_keys:
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    total_reference += len(value)
                    for item in value:
                        score_val = safe_int(item.get("match_score"))
                        if score_val > 0:
                            total_matches += 1
                elif isinstance(value, dict):
                    total_reference += 1
                    score_val = safe_int(value.get("match_score"))
                    if score_val > 0:
                        total_matches += 1

    df_unmatched_clean = calculate_clean_unmatched_criteria(df)
    total_unmatched = (df_unmatched_clean["clean_unmatched_inclusion_count"].sum() +
                       df_unmatched_clean["clean_unmatched_exclusion_count"].sum())

    precision = (total_matches / (total_matches + total_unmatched)
                 if (total_matches + total_unmatched) > 0 else float('nan'))
    recall = (total_matches / total_reference
              if total_reference > 0 else float('nan'))
    
    return precision, recall, total_matches, total_reference, total_unmatched

def extract_formatted_criteria(text):
    match = re.search(r'<FORMATTED_CRITERIA>(.*?)</FORMATTED_CRITERIA>', text, re.DOTALL)
    return match.group(1).strip() if match else ""

def calculate_batch_bertscores(df, batch_size=500):
    df['predicted_criteria'] = df['ec-raft-response'].apply(extract_formatted_criteria)
    
    mask = df['predicted_criteria'].str.len() > 0
    valid_df = df[mask].copy()
    
    all_precision = []
    all_recall = []
    all_f1 = []
    
    for i in tqdm(range(0, len(valid_df), batch_size), desc="Calculating BERTScores"):
        batch_pred = valid_df['predicted_criteria'][i:i+batch_size].tolist()
        batch_ref = valid_df['desired_criteria'][i:i+batch_size].tolist()
        assert len(batch_pred) == len(batch_ref)
        
        P, R, F1 = score(
            batch_pred, 
            batch_ref,
            verbose=False,
            batch_size=batch_size,
            model_type="distilbert-base-uncased",
            device='cuda'
        )
        
        all_precision.extend(P.tolist())
        all_recall.extend(R.tolist())
        all_f1.extend(F1.tolist())
    
    result_df = df.copy()
    result_df['bertscore_precision'] = float('nan')
    result_df['bertscore_recall'] = float('nan')
    result_df['bertscore_f1'] = float('nan')
    
    result_df.loc[mask, 'bertscore_precision'] = all_precision
    result_df.loc[mask, 'bertscore_recall'] = all_recall
    result_df.loc[mask, 'bertscore_f1'] = all_f1
    
    return result_df, all_precision, all_recall, all_f1

def calculate_weighted_stats(grouped_stats, df_parsed):
    total_count = grouped_stats['count'].sum()
    weighted_sum = (grouped_stats['count'] * grouped_stats['mean']).sum()
    weighted_avg = weighted_sum / total_count
    
    numerator = ((grouped_stats['count'] - 1) * (grouped_stats['std'] ** 2)).sum()
    denominator = (grouped_stats['count'] - 1).sum()
    pooled_std = np.sqrt(numerator / denominator)
    
    category_stats = {}
    for category in grouped_stats.index:
        cat_data = df_parsed[df_parsed['category'] == category]
        zero_matches = (cat_data['match_score'] == 0).sum()
        zero_match_percentage = (zero_matches / len(cat_data)) * 100
        
        category_stats[category] = {
            'count': int(grouped_stats.loc[category, 'count']),
            'mean': float(grouped_stats.loc[category, 'mean']),
            'std': float(grouped_stats.loc[category, 'std']),
            'zero_matches': int(zero_matches),
            'zero_match_percentage': float(zero_match_percentage)
        }
    
    total_zero_matches = (df_parsed['match_score'] == 0).sum()
    total_zero_percentage = (total_zero_matches / len(df_parsed)) * 100
    
    return weighted_avg, pooled_std, category_stats, total_zero_matches, total_zero_percentage

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of clinical trial criteria')
    parser.add_argument('--file', type=str, required=True, help='Input parquet file path')
    parser.add_argument('--exp', type=str, required=True, help='Experiment name')
    parser.add_argument('--reference_col', type=str, default='desired_criteria', help='Reference column name')
    parser.add_argument('--predicted_col', type=str, default='ec-raft-response', help='Predicted column name')
    parser.add_argument('--tool_model_path', type=str, default='watt-ai/watt-tool-8B', help='toolcall vLLM model path')
    parser.add_argument('--output_file', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    vertexai.init(project=PROJECT_ID, location="us-central1")
    
    # Create experiment directory
    exp_dir = f"./results/{args.exp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Loading data from {args.file}")
    df = pd.read_parquet(args.file)
    df = df.head(10)
    
    results = {
        'experiment': args.exp,
        'input_file': args.file,
        'total_samples': len(df)
    }
    
    print("Starting free text evaluation...")
    df = process_batch_free_text(df, args.reference_col, args.predicted_col, args.exp)
    if df is None:
        print("Free text evaluation failed")
        return
    df.to_parquet(f"{exp_dir}/{args.exp}_eval_complete.parquet")
    
    # Step 2: JSON parsing
    print("Starting JSON parsing...")
    df = process_json_parsing_with_vllm(df, args.exp, args.tool_model_path)
    df.to_parquet(f"{exp_dir}/{args.exp}_json_parsed.parquet")
    
    # Step 3: Parse and analyze results
    print("Analyzing results...")
    df = df.dropna(subset=["response_json"])
    
    # Parse JSON responses
    df_parsed = parse_json_from_df(df, "response_json")
    grouped_stats = df_parsed.groupby('category')['match_score'].describe()
    
    # Calculate weighted statistics
    weighted_avg, pooled_std, category_stats, total_zero_matches, total_zero_percentage = calculate_weighted_stats(grouped_stats, df_parsed)
    
    results['match_score_analysis'] = {
        'weighted_average': float(weighted_avg),
        'pooled_std': float(pooled_std),
        'total_zero_matches': int(total_zero_matches),
        'total_zero_percentage': float(total_zero_percentage),
        'category_statistics': category_stats
    }
    
    # Calculate precision and recall
    precision, recall, total_matches, total_reference, total_unmatched = compute_precision_recall_direct(df)
    
    results['precision_recall'] = {
        'precision': float(precision) if not np.isnan(precision) else None,
        'recall': float(recall) if not np.isnan(recall) else None,
        'total_matches': int(total_matches),
        'total_reference': int(total_reference),
        'total_unmatched': int(total_unmatched)
    }
    
    # Calculate BERTScores
    print("Calculating BERTScores...")
    df_with_scores, all_precision, all_recall, all_f1 = calculate_batch_bertscores(df)
    
    if all_f1:
        results['bertscore'] = {
            'precision_mean': float(np.mean(all_precision)),
            'precision_std': float(np.std(all_precision)),
            'recall_mean': float(np.mean(all_recall)),
            'recall_std': float(np.std(all_recall)),
            'f1_mean': float(np.mean(all_f1)),
            'f1_std': float(np.std(all_f1)),
            'valid_samples': len(all_f1)
        }
    
    # Calculate unmatched criteria statistics
    df_unmatched = calculate_clean_unmatched_criteria(df)
    results['unmatched_criteria'] = {
        'clean_unmatched_inclusion_mean': float(df_unmatched['clean_unmatched_inclusion_count'].mean()),
        'clean_unmatched_exclusion_mean': float(df_unmatched['clean_unmatched_exclusion_count'].mean()),
        'total_unmatched_inclusion_mean': float(df_unmatched['total_unmatched_inclusion_count'].mean()),
        'total_unmatched_exclusion_mean': float(df_unmatched['total_unmatched_exclusion_count'].mean())
    }
    
    output_file = args.output_file or f"{exp_dir}/{args.exp}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed. Results saved to {output_file}")
    print(f"Summary:")
    print(f"  Weighted Average Match Score: {weighted_avg:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    if all_f1:
        print(f"  BERTScore F1: {np.mean(all_f1):.4f}")
    
    return results

if __name__ == "__main__":
    main()
