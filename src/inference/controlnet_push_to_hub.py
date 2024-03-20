import os
from huggingface_hub import create_repo, upload_folder
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    output_dir = "controlnet-de-identification"
    hub_token = None

    repo_id = create_repo(
        repo_id=Path(output_dir).name, exist_ok=True, token=hub_token
    ).repo_id

    upload_folder(
        repo_id=repo_id,
        folder_path=output_dir,
        commit_message="End of training",
        ignore_patterns=["step_*", "epoch_*"],
    )
