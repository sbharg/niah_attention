from datasets import load_dataset
import pandas as pd
from huggingface_hub import hf_hub_download
import json

def load_mrcr_dataset(split="train"):
    dataset = load_dataset("openai/mrcr", split=split)
    return dataset

def load_mrcr_parquet():
    df = pd.read_parquet(
        hf_hub_download(repo_id="openai/mrcr", filename="2needle.parquet", repo_type="dataset")
    )
    return df

def parse_messages(prompt_str):
    return json.loads(prompt_str)
