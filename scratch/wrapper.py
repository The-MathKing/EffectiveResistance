import json
import os
import sys

def run():
    print("Running original test_cora...")
    # we can just run the test_cora.py script itself!
    os.system("PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE venv/bin/python src/test_cora.py > LATEST_CORA_OUTPUT.txt 2>&1")
    print("Done")

if __name__ == "__main__":
    run()
