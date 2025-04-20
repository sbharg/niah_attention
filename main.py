from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import MODEL_NAME, DEVICE, MAX_LENGTH
from mrcr_utils import load_mrcr_parquet, parse_messages
from data_preparation import format_prompt
from attention_utils import run_model_and_capture_attention, run_with_gradients
from visualizations import plot_attention_heatmap, plot_token_gradients, plot_multihead_attention
from evaluation import grade

def format_chatml_prompt(messages):
    input_text = "<|im_start|>system\nYou are a helpful assistant that follows instructions carefully.<|im_end|>\n"
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role in ["system", "user", "assistant"]:
            input_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    input_text += "<|im_start|>assistant\n" 
    return input_text

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        output_attentions=True,
        attn_implementation="eager",
        trust_remote_code=True, 
        return_dict_in_generate=True
    ).to(DEVICE)
    model.eval()

    dataset = load_mrcr_parquet()

    for i, row in dataset.iterrows():
        print(f"\n===== ROW {i + 1} =====")
        messages = parse_messages(row["prompt"])
        input_text = format_chatml_prompt(messages)
        #print("==== FULL PROMPT SENT TO MODEL ====")
        #print(input_text)
        #print("=" * 40)


        target = row["answer"]
        prefix = row["random_string_to_prepend"]

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)


        #outputs, attentions = run_model_and_capture_attention(model, inputs)
        print("Target:", target)
        #print("Finished attention heatmap.")
        #plot_attention_heatmap(attentions, tokenizer, inputs, target_phrase=target)
        #print("Finished multihead attention.")
        #plot_multihead_attention(attentions, tokenizer, inputs, layer=0)

        tokens, gradients = run_with_gradients(model, tokenizer, input_text, DEVICE)
        #print("Finished gradients.")

        #print("Plotting token gradients...")
        #plot_token_gradients(tokens, gradients)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=True
        )
        decoded = tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True)

        print("Generated:", decoded)
        print("Expected prefix:", prefix)
        print("Grading score:", grade(decoded, target, prefix))
        break 

if __name__ == "__main__":
    main()
