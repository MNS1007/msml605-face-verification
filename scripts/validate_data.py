import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.validation.checks import validate_all


def main():
    validate_all(
        train_pairs_path="outputs/pairs/train.csv",
        val_pairs_path="outputs/pairs/val.csv",
        test_pairs_path="outputs/pairs/test.csv",
        val_scored_path="outputs/scores/val_scored.csv",
        test_scored_path="outputs/scores/test_scored.csv",
        threshold=0.35,
    )
if __name__ == "__main__":
    main()