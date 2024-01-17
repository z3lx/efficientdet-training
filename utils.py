import os

def create_subdirectory(base_dir: str, prefix: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    sub_dir = None
    for i in range(len(os.listdir(base_dir)) + 1):
        sub_dir = os.path.join(base_dir, f"{prefix}_{i}")
        if not os.path.exists(sub_dir) or not os.listdir(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)
            break
    return sub_dir

def chdir_to_project_root():
    venv_dir = os.environ["VIRTUAL_ENV"]
    os.chdir(os.path.join(venv_dir, ".."))