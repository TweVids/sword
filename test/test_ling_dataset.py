import os
import json
import tempfile
from datasets import Dataset, load_dataset
from train_ling_example import load_and_prepare_dataset

def test_dataset_pipeline():
    # 1. Test with small sample from E:\lingtiny\merged_finetune_dataset_with_effort.jsonl
    source_file = r"E:\lingtiny\merged_finetune_dataset_with_effort.jsonl"
    if os.path.exists(source_file):
        with open(source_file, "r", encoding="utf-8") as f:
            sample_lines = [f.readline() for _ in range(5)]
        
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl", encoding="utf-8") as tmp:
            tmp.writelines(sample_lines)
            tmp_path = tmp.name

        try:
            ds = load_and_prepare_dataset(local_path=tmp_path, seed=42)
            assert len(ds) == 5, f"Expected 5 samples, got {len(ds)}"
            assert "messages" in ds[0], "Expected messages field in dataset"
            print("[PASS] Dataset sample parsed and shuffled successfully.")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        print("[SKIP] E:\\lingtiny dataset path not accessible.")

if __name__ == "__main__":
    test_dataset_pipeline()
