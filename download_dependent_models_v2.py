import os
from huggingface_hub import HfApi, hf_hub_download


# Create /opt directory if it doesn't exist
local_save_dir='my_models'

# Define the paths and model identifiers
model_identifiers={
    'SDXL_CKPT_PATH': 'stabilityai/stable-diffusion-xl-base-1.0',
    'LLAVA_CLIP_PATH': 'openai/clip-vit-large-patch14-336',
    'LLAVA_MODEL_PATH': 'liuhaotian/llava-v1.5-13b',
    'SDXL_CLIP1_PATH': 'openai/clip-vit-large-patch14',
    'SDXL_CLIP2_CKPT_PTH': 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
}

# Download and save the models
# Create the directory if it doesn't exist
os.makedirs(local_save_dir, exist_ok=True)

# Initialize the Hugging Face API
api = HfApi()

# Download and save all files from the models
for path_name, model_identifier in model_identifiers.items():
    print(f"Downloading files from {model_identifier}...")

    # Define the local save path
    local_path = os.path.join(local_save_dir, model_identifier.split("/")[-1])
    os.makedirs(local_path, exist_ok=True)

    # List all files in the repository
    files = api.list_repo_files(repo_id=model_identifier)

    # Download each file
    for file_name in files:
        print(f"Downloading {file_name}...")
        hf_hub_download(repo_id=model_identifier, filename=file_name, local_dir=local_path)

    print(f"All files from {model_identifier} saved to {local_path}")

