from transformers import AutoTokenizer
from datetime import datetime
import transformers
import torch


# Required to run:
# pip3 install transformers==4.31.0
# pip3 install accelerate
# pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121


def build_pipeline(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_dir,
        # torch_dtype=torch.float32,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return pipeline, tokenizer


def find_sequence(pipeline, tokenizer, question):
    print("Start Pipeline =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    sequences = pipeline(
        question,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    print("End Pipeline =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    print("Start Sequences =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
    print("End Sequences =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))


if __name__ == "__main__":
    # Model folder path and change this accordingly
    model_dir = "C:\model_huggingface\Llama-2-7b-chat-hf"

    # Define pipeline and tokenizer
    pipeline, tokenizer = build_pipeline(model_dir)

    # Ask question
    question = 'You are expert in fashion". Do you have any recommendations of jackets or coats when we are in winter season?'
    find_sequence(pipeline, tokenizer, question)

    # Ask question
    question = 'I liked "Breaking Bad". Do you have any recommendations of other shows I might like?'
    find_sequence(pipeline, tokenizer, question)