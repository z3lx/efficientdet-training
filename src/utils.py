import os

def create_subdirectory(base_dir: str, prefix: str) -> str:
    """Create a subdirectory with a given prefix in the given base directory"""
    os.makedirs(base_dir, exist_ok=True)
    sub_dir = None
    for i in range(len(os.listdir(base_dir)) + 1):
        sub_dir = os.path.join(base_dir, f"{prefix}_{i}")
        if not os.path.exists(sub_dir) or not os.listdir(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)
            break
    return sub_dir

def get_project_root(relative_path: str = None) -> str:
    """Get the root directory of the project"""
    venv_dir = os.environ["VIRTUAL_ENV"]
    relative_dir = os.path.join(venv_dir, "..", relative_path)
    os.makedirs(relative_dir, exist_ok = True)
    return relative_dir
