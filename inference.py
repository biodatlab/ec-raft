import os
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
from prompt_gen.inference_prompt_generator import InferencePromptGenerator


DEFAULT_MAX_MODEL_LEN = 25000
DEFAULT_NUM_GPUS = 4

def pipe(messages, llm, tokenizer, is_retry=False):
    sampling_params = SamplingParams(
        max_tokens=4096,
        min_p=0.03,
        temperature=0.3 if not is_retry else 0.4,
    )
    prompts = [tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False) for message in messages]
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]

class BatchProcessor:
    def __init__(self, llm, tokenizer, max_retries=3, batch_size=1000):
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.pending_retries = []
    
    def _process_single_batch(self, batch_indices, df, is_retry=False):
        messages_batch = [InferencePromptGenerator.create_messages(df.loc[idx, 'input']) for idx in batch_indices]
        results = pipe(messages_batch, self.llm, self.tokenizer, is_retry=is_retry)

        to_retry = []
        for idx, response in zip(batch_indices, results):
            if '</FORMATTED_CRITERIA>' not in response:
                print(response)
                if df.loc[idx, 'retry_count'] < self.max_retries:
                    df.loc[idx, 'retry_count'] += 1
                    to_retry.append(idx)
                    print(f"Response for index {idx} missing end tag; "
                          f"retry {df.loc[idx, 'retry_count']}/{self.max_retries}")
                else:
                    df.loc[idx, 'ec-raft-response'] = response
                    print(f"Max retries reached for index {idx}, saving partial response.")
            else:
                df.loc[idx, 'ec-raft-response'] = response

        return to_retry
    
    def _process_retry_queue(self, df):
        while self.pending_retries:
            retry_batch = self.pending_retries[:self.batch_size]
            self.pending_retries = self.pending_retries[self.batch_size:]
            new_retries = self._process_single_batch(retry_batch, df, is_retry=True)
            self.pending_retries.extend(new_retries)
    
    def process_dataframe(self, df, indices_to_process=None, checkpoint_callback=None):
        if indices_to_process is None:
            indices_to_process = df.index.tolist()
        
        with tqdm(total=len(indices_to_process), desc="Processing entries") as pbar:
            for i in range(0, len(indices_to_process), self.batch_size):
                if len(self.pending_retries) >= self.batch_size:
                    self._process_retry_queue(df)
                
                current_batch_indices = indices_to_process[i : i + self.batch_size]
                new_retries = self._process_single_batch(current_batch_indices, df, is_retry=False)
                self.pending_retries.extend(new_retries)
                
                if checkpoint_callback:
                    checkpoint_callback(i + self.batch_size, df)
                
                pbar.update(len(current_batch_indices))
            
            self._process_retry_queue(df)

def main():
    parser = argparse.ArgumentParser(description='Process studies with LLM')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Input parquet file path')
    parser.add_argument('--model_path', type=str,
                       help='Path to the model', default='biodatlab/ec-raft')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output parquet file path (default: derived from input file)')
    parser.add_argument('--max_model_len', type=int, default=DEFAULT_MAX_MODEL_LEN,
                       help='Maximum model length')
    parser.add_argument('--num_gpus', type=int, default=DEFAULT_NUM_GPUS,
                       help='Number of GPUs to use')
    
    args = parser.parse_args()

    if args.output_file is None:
        input_name = os.path.splitext(args.input_file)[0]
        args.output_file = f"{input_name}_inferred.parquet"

    intermediate_file = f"{os.path.splitext(args.output_file)[0]}_checkpoint.parquet"

    llm = LLM(model=args.model_path, 
              tensor_parallel_size=args.num_gpus, 
              max_model_len=args.max_model_len)
    tokenizer = llm.get_tokenizer()

    if os.path.exists(intermediate_file):
        df = pd.read_parquet(intermediate_file)
        indices_to_process = df[df['ec-raft-response'].isna()].index.tolist()
        print(f"Resuming from checkpoint: {len(indices_to_process)} entries remaining to process")
    else:
        df = pd.read_parquet(args.input_file)
        df = df.dropna()
        indices_to_process = None

    if 'ec-raft-response' not in df.columns:
        df['ec-raft-response'] = None
    if 'retry_count' not in df.columns:
        df['retry_count'] = 0

    df['input'] = df.apply(
        lambda x: InferencePromptGenerator.create_input(
            x['related_studies_context'],
            x['title'],
            x['description']
        ),
        axis=1
    )

    def save_checkpoint(current_idx, dataframe):
        dataframe.to_parquet(intermediate_file)

    batch_processor = BatchProcessor(
        llm=llm,
        tokenizer=tokenizer,
        max_retries=3,
        batch_size=1000
    )
    
    batch_processor.process_dataframe(df, indices_to_process, save_checkpoint)

    if os.path.exists(intermediate_file):
        os.remove(intermediate_file)

    df.to_parquet(args.output_file)
    print(f"Processing complete. Final results saved to '{args.output_file}'")

if __name__ == "__main__":
    main() 