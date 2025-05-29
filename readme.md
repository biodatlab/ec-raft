# EC-RAFT: Automated Generation of Clinical Trial Eligibility Criteria through Retrieval-Augmented Fine-Tuning

This repository contains the implementation of EC-RAFT, a method that utilizes Retrieval-Augmented Fine-Tuning (RAFT) to generate structured and cohesive eligibility criteria directly from clinical trial titles and descriptions.

## 🚀 Installation

### Setup

1. Clone the repository:
```bash
git clone https://github.com/biodatlab/ec-raft
cd ec-raft
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install XTuner for fine-tuning:
```bash
pip install xtuner
```

## 🏃 Quick Start

### 1. Data Embedding

First, create embeddings for the clinical trials dataset:

```bash
python embed.py
```

This script:
- Loads the clinical trials dataset from HuggingFace
- Creates embeddings using the SciNCL model
- Stores embeddings in ChromaDB for retrieval

### 2. Data Preprocessing

Generate training data with intermediate reasoning steps:

```bash
python preprocess.py --experiment_name "my_experiment" --top_n 4 --split train
```

Parameters:
- `--experiment_name`: Name for your experiment
- `--top_n`: Number of similar trials to retrieve (1-5)
- `--split`: Dataset split to process (train/validation/test)

### 3. Fine-tuning

Configure and run fine-tuning using XTuner:

```bash
# Edit the configuration file
vim config/ec-raft-xtuner.py

# Update the following paths in the config:
# - pretrained_model_name_or_path: Path to your base model (e.g., Llama-3.1-8B-Instruct)
# - cache_dir: Directory for caching models
# - ecraft_en_path: Path to your processed training data

# Run fine-tuning
xtuner train config/ec-raft-xtuner.py
```

### 4. Evaluation

Evaluate your fine-tuned model:

```bash
python evaluate.py --experiment_name "my_experiment" --model_path "watt-ai/watt-tool-8B"
```


### Preprocessing Pipeline

1. **Embedding Creation** (`embed.py`):
   - Processes clinical trials from the dataset
   - Creates embeddings using SciNCL
   - Stores in ChromaDB for retrieval

2. **Data Generation** (`preprocess.py`):
   - Retrieves similar trials for each target trial
   - Generates intermediate reasoning steps using LLMs
   - Creates training datasets with different top-N configurations

## 🔧 Fine-tuning

### Configuration

The fine-tuning process uses XTuner with LoRA for efficient training. Key configuration parameters in `config/ec-raft-xtuner.py`:

```python
# Model settings
pretrained_model_name_or_path = '<PATH_TO_MODEL>'  # e.g., "meta-llama/Llama-3.1-8B-Instruct"
cache_dir = "<PATH_TO_CACHE>"

# Data settings
ecraft_en_path = '<PATH_TO_DATA>'  # Path to your processed training data 
```

### Running Fine-tuning

1. **Prepare your data**: Ensure your training data is in the correct format
2. **Update configuration**: Modify paths in `config/ec-raft-xtuner.py`
3. **Start training**:
   ```bash
   xtuner train config/ec-raft-xtuner.py
   ```

- **GPU Requirements**: 4x NVIDIA A100 GPUs

## 📈 Evaluation

### Metrics

The evaluation pipeline includes:

1. **BERTScore**: Semantic similarity between generated and reference criteria
2. **LLM-Guided Evaluation**: Clinical relevance assessment using LLM-as-a-Judge
3. **Precision/Recall**: Quantitative agreement metrics

### Running Evaluation

```bash
python evaluate.py \
    --experiment_name "my_experiment" \
    --tool_model_path "path/to/watt-tool-8B" \
    --reference_column "desired_criteria" \
    --predicted_column "response"
```

### Evaluation Process

1. **Free-text Evaluation**: Uses Gemini to assess clinical relevance
2. **Structured Parsing**: Converts evaluations to JSON format
3. **Metric Calculation**: Computes BERTScore, precision, recall, and judge scores
