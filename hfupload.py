import torch
from pathlib import Path
from huggingface_hub import HfApi, upload_file
import json

repo_id = "Sarthak2910/transformer-en-hi"
model_folder = Path("iitb_weights") 
config = {
    "seq_len": 128,
    "d_model": 512,
    "lang_src": "en",
    "lang_tgt": "hi"
}


api = HfApi()
api.create_repo(repo_id=repo_id, exist_ok=True)

weight_files = sorted(model_folder.glob("*.pt"))
if not weight_files:
    raise Exception("No model weights found!")

latest_model_path = weight_files[-1]
print(f"Using model: {latest_model_path}")

state = torch.load(latest_model_path, map_location="cpu")

final_model_path = "pytorch_model.pt"
torch.save(state["model_state_dict"], final_model_path)

with open("config.json", "w") as f:
    json.dump(config, f, indent=4)

files_to_upload = [
    final_model_path,
    "config.json",
    "tokenizer_en.json",
    "tokenizer_hi.json",
]

for file in files_to_upload:
    path = Path(file)
    if path.exists():
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
        )
        print(f"Uploaded: {file}")
    else:
        print(f"Skipped (not found): {file}")

print("Your model is pushed to hf successfully ")
