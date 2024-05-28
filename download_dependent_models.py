import os
from transformers import CLIPProcessor, CLIPModel

# Create /opt directory if it doesn't exist
os.makedirs('my_models', exist_ok=True)

# Define the paths and model identifiers
model_identifiers = {
    'LLAVA_CLIP_PATH': 'openai/clip-vit-large-patch14-336',
    'LLAVA_MODEL_PATH': 'liuhaotian/llava-v1.5-13b',
    'SDXL_CLIP1_PATH': 'openai/clip-vit-large-patch14',
    'SDXL_CLIP2_CKPT_PTH': 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
}

# Download and save the models
for path_name, model_identifier in model_identifiers.items():
    print(f"Downloading {model_identifier}...")
    
    model = CLIPModel.from_pretrained(model_identifier)
    processor = CLIPProcessor.from_pretrained(model_identifier)

    # Define the local save path
    local_path = f'my_models/{model_identifier.split("/")[-1]}'
    model.save_pretrained(local_path)
    processor.save_pretrained(local_path)

    print(f"Model {model_identifier} saved to {local_path}")


os.system('sudo mv models/* /opt/')
