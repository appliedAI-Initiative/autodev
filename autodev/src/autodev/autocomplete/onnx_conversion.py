import logging
from pathlib import Path

import transformers
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import PreTrainedModel
from transformers.onnx import FeaturesManager

log = logging.getLogger(__name__)


class ONNXConversion:
    def __init__(self, opset_version=13):
        self.opset_version = opset_version

    def convert(self, model: PreTrainedModel, tokenizer, output_dir: Path):
        """
        :param model: the model to convert
        :param output_dir: the path to a directory in which the converted model shall be stored. Note that multiple
            files will be written. If the directory does not exist, it will be created.
        """
        log.info("Checking model features")
        feature = "causal-lm"
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
        onnx_config = model_onnx_config(model.config)

        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = output_dir / "model.onnx"

        log.info("Performing ONNX conversion ...")
        transformers.onnx.export(
            preprocessor=tokenizer,
            model=model,
            config=onnx_config,
            opset=self.opset_version,
            output=onnx_path
        )

        model.config.save_pretrained(output_dir)

    @staticmethod
    def _quantize(onnx_src_dir: Path, onnx_target_dir: Path, qconfig):
        """
        :param onnx_src_dir: source directory containing one or more .onnx models to be quantized
        :param onnx_target_dir: target directory in which to save the quantized models to be generated
        :param qconfig: the quantization configuration
        """
        for onnx_file_path in onnx_src_dir.glob("*.onnx"):
            log.info(f"Quantizing {onnx_file_path}")
            quantizer = ORTQuantizer.from_pretrained(onnx_src_dir, file_name=onnx_file_path.name)
            quantizer.quantize(qconfig, save_dir=onnx_target_dir)

    @classmethod
    def quantize_avx512(cls, onnx_src_dir: Path, onnx_target_dir: Path):
        """
        :param onnx_src_dir: source directory containing one or more .onnx models to be quantized
        :param onnx_target_dir: target directory in which to save the quantized models to be generated
        """
        # NOTE: This is analogous to the CLI command:
        # optimum-cli onnxruntime quantize --avx512 --onnx_model src_dir -o target_dir
        cls._quantize(onnx_src_dir, onnx_target_dir, AutoQuantizationConfig.avx512(is_static=False,
            per_channel=False))

    @classmethod
    def quantize_avx512_vnni(cls, onnx_src_dir: Path, onnx_target_dir: Path):
        """
        :param onnx_src_dir: source directory containing one or more .onnx models to be quantized
        :param onnx_target_dir: target directory in which to save the quantized models to be generated
        """
        cls._quantize(onnx_src_dir, onnx_target_dir, AutoQuantizationConfig.avx512_vnni(is_static=False,
            per_channel=False))
