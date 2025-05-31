# EC-RAFT: Automated Generation of Clinical Trial Eligibility Criteria through Retrieval-Augmented Fine-Tuning

This repository contains the implementation of EC-RAFT (ACL 2025 findings), a method that utilizes Retrieval-Augmented Fine-Tuning (RAFT) to generate structured and cohesive **eligibility criteria** directly from clinical trial titles and descriptions.

## 🚀 Installation

Start by cloning the repository and installing the required dependencies:

```bash
git clone https://github.com/biodatlab/ec-raft
cd ec-raft
pip install -r requirements.txt
pip install xtuner
```

## Predict EC

TBD

## 🏃 Quick Start

The EC-RAFT method follows a systematic preprocessing pipeline. The embedding creation step processes clinical trials from the dataset and uses the SciNCL model to create semantic embeddings, which are then stored in ChromaDB for fast retrieval. During data generation, the system retrieves similar trials for each target trial and uses large language models to generate intermediate reasoning steps, creating comprehensive training datasets.

### Step 1: Create Data Embeddings

Generate embeddings for the clinical trials dataset using the SciNCL model:

```bash
python embed.py
```

This script loads the clinical trials dataset from HuggingFace, creates embeddings, and stores them in ChromaDB for efficient retrieval during training.

### Step 2: Preprocess Training Data

Generate training data with intermediate reasoning steps by retrieving similar trials:

```bash
python preprocess.py --experiment_name "my_experiment" --top_n 4 --split train
```

The `--top_n` parameter controls how many similar trials to retrieve (1-5), while `--experiment_name` sets your experiment identifier. You can process different dataset splits using the `--split` parameter (train/validation/test).

### Step 3: Configure and Run Fine-tuning

Edit the [XTuner](https://github.com/InternLM/xtuner) configuration file to set your model and data paths.

```bash
# Model settings
pretrained_model_name_or_path = '<PATH_TO_MODEL>'  # e.g., "meta-llama/Llama-3.1-8B-Instruct"
cache_dir = "<PATH_TO_CACHE>"

# Data settings
ecraft_en_path = '<PATH_TO_DATA>'  # Path to your processed training data 
```

The fine-tuning process leverages XTuner with LoRA (Low-Rank Adaptation) for efficient training.
Ensure your training data is in the correct format. **GPU Requirements**: 4x NVIDIA A100 GPUs.
Then start training:

```bash
# Run fine-tuning
xtuner train config/ec-raft-xtuner.py
```


### 4. Evaluation

Run evaluation on your fine-tuned model:

```bash
python evaluate.py --experiment_name "my_experiment" --model_path "watt-ai/watt-tool-8B"
```


## 📈 Evaluation

**BERTScore** measures semantic similarity between generated and reference criteria, while **LLM-guided evaluation** uses a judge model to assess clinical relevance. Traditional **precision** and **recall** metrics provide quantitative agreement measurements. To run a complete evaluation:

```bash
python evaluate.py \
    --experiment_name "my_experiment" \
    --tool_model_path "path/to/watt-tool-8B" \
    --reference_column "desired_criteria" \
    --predicted_column "response"
```

The evaluation process includes free-text assessment using Gemini for clinical relevance, structured parsing to convert evaluations into JSON format, and metric calculation covering BERTScore, precision, recall, and judge scores.

1. **Free-text Evaluation**: Uses Gemini to assess clinical relevance
2. **Structured Parsing**: Converts evaluations to JSON format
3. **Metric Calculation**: Computes BERTScore, precision, recall, and judge scores
