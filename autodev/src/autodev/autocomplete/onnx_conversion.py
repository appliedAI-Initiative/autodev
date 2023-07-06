import logging
from pathlib import Path

import transformers
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
