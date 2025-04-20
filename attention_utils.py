import torch

def run_model_and_capture_attention(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)
        attentions = outputs.attentions  # list of tensors (layers, batch, head, query, key)
    return outputs, attentions

def run_with_gradients(model, tokenizer, input_text, device):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
    inputs_embeds.retain_grad()

    outputs = model(inputs_embeds=inputs_embeds)
    loss = outputs.logits.sum()
    loss.backward()

    gradients = inputs_embeds.grad[0].mean(dim=1).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return tokens, gradients
