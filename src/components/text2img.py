import os
import numpy as np
from transformers import CLIPTokenizer, FlaxCLIPTextModel


class CLIPEmbedder:
    def __init__(self, max_length, skip_layers: int = 0, hf_model: str = "openai/clip-vit-large-patch14"):
        self.tokenizer = CLIPTokenizer.from_pretrained(hf_model)
        self.model = FlaxCLIPTextModel.from_pretrained(
            hf_model, num_hidden_layers=12 - (skip_layers - 1))
        self.max_length = max_length

    def embed_prompts(self, prompts: list[str]):
        batch_encoding = self.tokenizer(prompts, truncation=True, max_length=self.max_length,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="jax")

        batch_encoding = {key: batch_encoding[key]
                          for key in batch_encoding if key != 'length'}

        outputs = self.model(**batch_encoding)

        z = outputs.last_hidden_state
        return z

    def get_empty_context(self):
        return self.embed_prompts([""])[0]


def process_directory(input_dir: str, output_dir: str, max_length: int, skip_layers: int = 0, hf_model: str = "openai/clip-vit-large-patch14"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    embedder = CLIPEmbedder(max_length, skip_layers, hf_model)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):  # Assuming the prompts are stored in .txt files
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r") as file:
                prompts = [line.strip() for line in file.readlines()]

            embeddings = embedder.embed_prompts(prompts)
            embeddings_np = np.array(embeddings)  # Convert to numpy array

            # Save embeddings to numpy file
            output_file_path = os.path.join(
                output_dir, filename.replace(".txt", ".npy"))
            np.save(output_file_path, embeddings_np)


if __name__ == "__main__":
    input_directory = "./dataset_prompts"
    output_directory = "./embeddings_output"
    max_length = 77

    process_directory(input_directory, output_directory, max_length)
