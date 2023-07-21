from pathlib import Path

from autodev.util import logging
from autodev.autocomplete.onnx_conversion import ONNXConversion

log = logging.getLogger(__name__)


def convert_single(input_model_path: Path, avx512=True, vnni=True):
    if avx512:
        target_path = input_model_path.with_name(input_model_path.name + "_avx512")
        ONNXConversion.quantize_avx512(input_model_path, target_path)
    if vnni:
        target_path = input_model_path.with_name(input_model_path.name + "_avx512-vnni")
        ONNXConversion.quantize_avx512_vnni(input_model_path, target_path)


def convert_dual(input_model_path):
    target_dir = input_model_path.with_name(input_model_path.name + "_avx512")
    for dir in [ONNXConversion.SUBMODEL_DIR_WITHOUT_PAST, ONNXConversion.SUBMODEL_DIR_WITH_PAST]:
        ONNXConversion.quantize_avx512(input_model_path / dir, target_dir / dir)


if __name__ == '__main__':
    logging.configure()
    log.info("Starting")

    # apply quantization to previously created ONNX model (see run_onnx_conversion)
    #convert_single(Path("models/santacoder_onnx"))
    convert_dual(Path("models/santacoder_onnx_cached"))

    log.info("Done")
