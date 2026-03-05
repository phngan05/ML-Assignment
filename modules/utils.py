import os

def setup_data(dataset_slug, target_dir="data"):
    # 1. Check if data already exists to avoid redundant downloads
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"--- Data already exists at '{target_dir}'. Skipping download. ---")
        return

    print("--- Setting up Kaggle API and downloading data... ---")
    
    # 2. Set Kaggle API credentials (replace with your actual credentials)
    os.environ['KAGGLE_USERNAME'] = "phamhongngan"
    os.environ['KAGGLE_KEY'] = "9ad4b995a657d8cc450bcb44b1956724"

    # 3. Create directory and download data
    os.makedirs(target_dir, exist_ok=True)
    command = f"kaggle datasets download -d {dataset_slug} -p {target_dir} --unzip"
    os.system(command)
    print(f"--- Data downloaded successfully into '{target_dir}' ---")
