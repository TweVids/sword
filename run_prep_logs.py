"""Run sword setup + diagnostics (prep_and_logs) exactly as documented in train_entry."""

# Activate the repo venv first (Windows):
#   venv\Scripts\activate
# Or run directly with the venv python:
#   venv\Scripts\python.exe run_prep_logs.py
import os
import sword

result = sword.train(
    prep_and_logs=True,   # setup provisioning + full diagnostics log run
    setup=True,           # provision/validate environment before checking
    test=True,            # run the live inference test suite
    b_model=True,         # include Model B judge (ollama qwen3.6:35b @ port 11343)
    # Optional knobs (defaults shown):
    # model=None,                 # default in this mode: google/gemma-4-31B-it (8-bit GRPO path)
    # B_model_port=11343,
    # HF_TOKEN=os.environ.get("HF_TOKEN", ""),  # or pass token directly
    # mock_test=True,             # <- on a CPU-only box, set this to skip heavy GPU loads
)

print("ok:", result.get("ok"))
print("status:", result.get("status"))
print(result)
