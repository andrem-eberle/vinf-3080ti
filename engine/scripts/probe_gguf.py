from __future__ import annotations

import argparse

from vinf.gguf.parser import load_gguf
from vinf.gguf.qwen import probe_qwen_gguf


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe GGUF metadata without reading tensor payloads.")
    parser.add_argument("path")
    args = parser.parse_args()
    gguf = load_gguf(args.path)
    report = probe_qwen_gguf(gguf)
    print(f"architecture={report.architecture}")
    print(f"tensors={report.tensor_count}")
    print(f"metadata={report.metadata_count}")
    print(f"layers={report.metadata.num_hidden_layers}")
    print(f"hidden_size={report.metadata.hidden_size}")
    print(f"vocab_size={report.metadata.vocab_size}")
    print(f"dtype={report.metadata.dtype}")
    print(f"quantized_tensors={report.quantized_tensor_count}")
    print("tensor_types=" + ",".join(f"{k}:{v}" for k, v in sorted(report.tensor_type_counts.items())))
    if report.feature_fields_present:
        print("feature_fields=" + ",".join(report.feature_fields_present))
    print(f"supported_for_inference={str(report.supported_for_inference).lower()}")
    for reason in report.unsupported_reasons:
        print(f"unsupported={reason}")


if __name__ == "__main__":
    main()
