import chromadb
from sentence_transformers import SentenceTransformer
from client import Client
from prompt_gen.inference_prompt_generator import InferencePromptGenerator
from vllm import LLM, SamplingParams
import re

# Initialize ChromaDB client and embedding model
client = chromadb.PersistentClient(path="./clinical_trials_chroma_all")
model = SentenceTransformer("malteos/scincl", device="cpu")
collection = client.get_or_create_collection("clinical_trials_studies")

# Initialize EC-RAFT client and prompt generator
ec_raft_client = Client(client, model, collection)
prompt_generator = InferencePromptGenerator(ec_raft_client)

# Initialize VLLM model
llm = LLM(model="biodatlab/ec-raft", max_model_len=13000)
sampling_params = SamplingParams(temperature=0.3, min_p=0.03, top_p=0.95, max_tokens=4096)

def generate_ec(title: str, description: str):
    """
    Generates eligibility criteria based on a title and description.

    Args:
        title (str): The title of the study.
        description (str): The description of the study.

    Returns:
        dict: A dictionary containing 'reasoning', 'criteria', 'raw_output', and 'prompt'.
    """
    if not title.strip() and not description.strip():
        return {
            "reasoning": "Input is empty.",
            "criteria": "",
            "raw_output": "",
            "prompt": ""
        }

    messages = prompt_generator.generate_inference_messages(title, description)
    prompt = llm.get_tokenizer().apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    outputs = llm.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text
    
    # Separate the reasoning from the formatted criteria
    reasoning = response.split('<FORMATTED_CRITERIA>')[0].strip()
    formatted_criteria_match = re.search(r"<FORMATTED_CRITERIA>(.*)</FORMATTED_CRITERIA>", response, re.DOTALL)
    
    criteria_block = ""
    if formatted_criteria_match:
        criteria_block = formatted_criteria_match.group(1).strip()

    return {
        "reasoning": reasoning,
        "criteria": criteria_block,
        "raw_output": response,
        "prompt": prompt
    }
