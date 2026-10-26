import kagglehub

def setup_data(dataset_slug, target_dir="data"):
    path = kagglehub.dataset_download(dataset_slug, output_dir=target_dir)

    print("Path to dataset files:", path)
