from pathlib import Path

from autodev.util import logging
from autodev.autocomplete.onnx_conversion import ONNXConversion

log = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.configure()
    log.info("Starting")

    # apply quantization to previously created ONNX model (see run_onnx_conversion)
    model_name = "santacoder_onnx"
    input_model_path = Path(f"models/{model_name}")
    #ONNXConversion.quantize_avx512(input_model_path, Path(f"models/{model_name}_avx512"))
    ONNXConversion.quantize_avx512_vnni(input_model_path, Path(f"models/{model_name}_avx512-vnni"))

    log.info("Done")
