from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from vinf.errors import UnsupportedModelError
from vinf.gguf.parser import GGUFFile, GGUFTensorType
from vinf.models.metadata import ModelMetadata
from vinf.gguf.mapper import metadata_from_gguf


SUPPORTED_FIRST_QWEN_ARCHITECTURES = {"qwen35"}
QWEN35_FEATURE_FIELDS = (
    "qwen35.ssm.conv_kernel",
    "qwen35.ssm.group_count",
    "qwen35.ssm.inner_size",
    "qwen35.ssm.state_size",
    "qwen35.ssm.time_step_rank",
    "qwen35.full_attention_interval",
    "qwen35.nextn_predict_layers",
)


@dataclass(frozen=True, slots=True)
class QwenGGUFReadinessReport:
    metadata: ModelMetadata
    architecture: str
    tensor_count: int
    metadata_count: int
    quantized_tensor_count: int
    tensor_type_counts: dict[str, int]
    feature_fields_present: tuple[str, ...]
    unsupported_reasons: tuple[str, ...]

    @property
    def supported_for_inference(self) -> bool:
        return not self.unsupported_reasons


def probe_qwen_gguf(gguf: GGUFFile) -> QwenGGUFReadinessReport:
    architecture = str(gguf.metadata_value("general.architecture", ""))
    metadata = metadata_from_gguf(gguf)
    tensor_type_counts = Counter(tensor.tensor_type.name for tensor in gguf.tensors.values())
    feature_fields = tuple(
        field for field in QWEN35_FEATURE_FIELDS if gguf.metadata_value(field) is not None
    )
    reasons: list[str] = []
    if architecture not in SUPPORTED_FIRST_QWEN_ARCHITECTURES:
        reasons.append(f"unsupported Qwen architecture: {architecture}")
    unsupported_quant = sorted(
        name
        for name in tensor_type_counts
        if GGUFTensorType[name] not in {
            GGUFTensorType.F32,
            GGUFTensorType.F16,
            GGUFTensorType.Q8_0,
            GGUFTensorType.Q2_K,
            GGUFTensorType.Q4_K,
            GGUFTensorType.Q5_K,
            GGUFTensorType.Q6_K,
            GGUFTensorType.Q3_K,
            GGUFTensorType.IQ4_XS,
            GGUFTensorType.IQ4_NL,
            GGUFTensorType.IQ3_S,
            GGUFTensorType.IQ2_XXS,
            GGUFTensorType.IQ2_XS,
            GGUFTensorType.IQ2_S,
            GGUFTensorType.IQ3_XXS,
        }
    )
    if unsupported_quant:
        reasons.append("unsupported tensor types: " + ", ".join(unsupported_quant))
    return QwenGGUFReadinessReport(
        metadata=metadata,
        architecture=architecture,
        tensor_count=len(gguf.tensors),
        metadata_count=len(gguf.metadata),
        quantized_tensor_count=len(gguf.quantized_tensors()),
        tensor_type_counts=dict(tensor_type_counts),
        feature_fields_present=feature_fields,
        unsupported_reasons=tuple(reasons),
    )


def require_qwen_inference_ready(gguf: GGUFFile) -> QwenGGUFReadinessReport:
    report = probe_qwen_gguf(gguf)
    if not report.supported_for_inference:
        raise UnsupportedModelError("; ".join(report.unsupported_reasons))
    return report
