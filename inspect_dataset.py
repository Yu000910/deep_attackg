"""inspect_dataset.py - BEDR resampled dataset distribution statistics.

Reads BEDR_resampled_dataset.csv and prints the class-count distribution statistics
reported in the manuscript (Table 1 footnote):
  - Total samples, distinct classes
  - Min, max, mean, median, standard deviation
  - Distribution buckets: <=10, 11-20, 21-30, 31-40, 41-50, 51-60, 61-70, 71-80

Usage:
    python inspect_dataset.py [--csv BEDR_resampled_dataset.csv]
"""

import sys
import os
import pandas as pd
import numpy as np

DEFAULT_CSV = "BEDR_resampled_dataset.csv"


def inspect(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        print("Usage: python inspect_dataset.py [path/to/BEDR_resampled_dataset.csv]")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Determine the technique column name
    if "technique" in df.columns:
        col = "technique"
    elif "tech_id" in df.columns:
        col = "tech_id"
    else:
        print(f"Error: Could not find technique/tech_id column. Columns: {list(df.columns)}")
        sys.exit(1)

    counts = df[col].value_counts()
    total = len(df)
    n_classes = len(counts)

    print("=" * 60)
    print(f"Dataset: {csv_path}")
    print("=" * 60)
    print(f"  Total samples:       {total}")
    print(f"  Distinct classes:    {n_classes}")
    print(f"  Min samples/class:   {counts.min()}")
    print(f"  Max samples/class:   {counts.max()}")
    print(f"  Mean:                {counts.mean():.1f}")
    print(f"  Median:              {counts.median():.1f}")
    print(f"  Standard deviation:  {counts.std():.1f}")
    print()

    # Distribution buckets
    buckets = [
        ("<=10", lambda c: c <= 10),
        ("11-20", lambda c: 11 <= c <= 20),
        ("21-30", lambda c: 21 <= c <= 30),
        ("31-40", lambda c: 31 <= c <= 40),
        ("41-50", lambda c: 41 <= c <= 50),
        ("51-60", lambda c: 51 <= c <= 60),
        ("61-70", lambda c: 61 <= c <= 70),
        ("71-80", lambda c: 71 <= c <= 80),
    ]
    print("  Distribution buckets:")
    for label, pred in buckets:
        n = counts[pred].count()
        print(f"    {label:>6}: {n} classes")
    print()

    # Classes at the ceiling (80)
    at_ceiling = (counts == 80).sum()
    print(f"  Classes at ceiling (80 samples): {at_ceiling}")

    # Verify total matches n_classes
    total_bucketed = sum(counts[pred].count() for _, pred in buckets)
    assert total_bucketed == n_classes, f"Bucket sum {total_bucketed} != {n_classes}"
    print(f"  Bucket coverage: {total_bucketed}/{n_classes} classes  (OK)")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    inspect(csv_path)
