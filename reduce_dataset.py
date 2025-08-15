import random
import os

def sample_fasttext_file(input_path, output_path, sample_size=5000, random_seed=42):
    """
    Randomly samples 'sample_size' lines from a fastText format .txt file and saves them to output_path.
    If sample_size is larger than the size of the dataset, copies the entire file.
    """
    random.seed(random_seed)
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if sample_size >= len(lines):
        print(f"Sample size {sample_size} >= data size {len(lines)}; copying entire dataset.")
        sample_lines = lines
    else:
        sample_lines = random.sample(lines, sample_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(sample_lines)

    print(f"Saved {len(sample_lines)} samples to '{output_path}'")

if __name__ == "__main__":
    # Update these paths to your actual locations
    input_train = r"C:\Users\poola\Downloads\Sentiment analysis\train.ft.txt"
    output_train = r"C:\Users\poola\Downloads\Sentiment analysis\train_small.ft.txt"

    input_test = r"C:\Users\poola\Downloads\Sentiment analysis\test.ft.txt"
    output_test = r"C:\Users\poola\Downloads\Sentiment analysis\test_small.ft.txt"

    sample_train_size = 5000    # Number of training samples you want
    sample_test_size = 1000     # Number of test samples you want

    sample_fasttext_file(input_train, output_train, sample_train_size)
    sample_fasttext_file(input_test, output_test, sample_test_size)