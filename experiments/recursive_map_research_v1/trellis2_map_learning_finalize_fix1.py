"""Normalize the existing occupancy receipt key; do not modify receipts."""
import trellis2_map_learning_finalize_v1 as report
from pathlib import Path

read_original = report.read


def read_receipt_compatible(path):
    record = read_original(path)
    if Path(path).parent.name == 'gpu_cost':
        record['actual_GPU_seconds'] = record['gpu_seconds']
    return record


if __name__ == '__main__':
    report.read = read_receipt_compatible
    report.main()
