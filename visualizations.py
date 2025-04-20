import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_attention_heatmap(attentions, tokenizer, inputs, target_phrase=""):
    layer = -1  # last layer
    head = 0  # first head
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    attn_matrix = attentions[layer][0, head].cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn_matrix, aspect='auto', cmap="viridis")

    ax.set_title(f"Layer {layer}, Head {head} Attention")
    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def plot_token_gradients(tokens, gradients):
    plt.figure(figsize=(12, 6))
    plt.plot(gradients, marker='o')
    plt.title("Averaged Gradients for Input Tokens")
    plt.xlabel("Token Index")
    plt.ylabel("Average Gradient Value")
    plt.xticks(ticks=range(len(gradients)), labels=tokens, rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def plot_multihead_attention(attentions, tokenizer, inputs, layer=0):
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    num_heads = attentions[layer].shape[1]
    fig, axes = plt.subplots(1, num_heads, figsize=(20, 5))
    for i in range(num_heads):
        sns.heatmap(attentions[layer][0, i].detach().cpu().numpy(),
                    xticklabels=tokens,
                    yticklabels=tokens,
                    cmap="viridis",
                    ax=axes[i])
        axes[i].set_title(f"Head {i+1}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)
    plt.close()
