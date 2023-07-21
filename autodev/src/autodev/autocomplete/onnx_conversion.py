import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Mapping, Any, List

import transformers
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, TensorType, is_torch_available, PretrainedConfig
from transformers.models.gpt2 import GPT2OnnxConfig
from transformers.onnx import FeaturesManager, OnnxConfigWithPast, PatchingSpec, OnnxConfig

log = logging.getLogger(__name__)


class SantaCoderOnnxConfig(GPT2OnnxConfig):
    def __init__(
            self,
            config: PretrainedConfig,
            task: str = "default",
            patching_specs: List[PatchingSpec] = None,
            use_past: bool = False,
            use_past_outputs: bool = False
    ):
        super().__init__(config, task, patching_specs, use_past=use_past)
        self.use_past_outputs = use_past_outputs

    def generate_dummy_inputs(
            self,
            tokenizer: PreTrainedTokenizer,
            batch_size: int = -1,
            seq_length: int = -1,
            is_pair: bool = False,
            framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        common_inputs = OnnxConfigWithPast.generate_dummy_inputs(self,
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # We need to order the input in the way they appears in the forward()
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # Need to add the past_keys
        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # Not using the same length for past_key_values
                past_length = 2  # any value
                past_key_shape = (
                    batch,
                    self._config.hidden_size // self.num_attention_heads,
                    past_length,
                )
                past_values_shape = (
                    batch,
                    past_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                ordered_inputs["past_key_values"] = [
                    (torch.rand(past_key_shape), torch.rand(past_values_shape)) for _ in range(self.num_layers)
                ]

        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.ones(batch, seqlen + past_length, dtype=mask_dtype)
        return ordered_inputs

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = OnnxConfigWithPast.outputs.fget(self)
        if self.use_past or self.use_past_outputs:
            self.fill_with_past_key_values_(common_outputs, direction="outputs")

        return common_outputs

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        if hasattr(self._config, "use_cache"):
            return {"use_cache": self.use_past or self.use_past_outputs}

        return None

    def fill_with_past_key_values_(
            self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str, inverted_values_shape: bool = False
    ):
        # must force inverted value shapes for santacoder
        super().fill_with_past_key_values_(inputs_or_outputs, direction, inverted_values_shape=True)


class ONNXConversion:
    TASK = "causal-lm"
    SUBMODEL_DIR_WITHOUT_PAST = "model_without_past"
    SUBMODEL_DIR_WITH_PAST = "model_with_past"

    def __init__(self, opset_version=13):
        self.opset_version = opset_version

    def _convert(self, model: PreTrainedModel, onnx_config: OnnxConfig, tokenizer, output_dir: Path):
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

    def convert(self, model: PreTrainedModel, tokenizer, output_dir: Path):
        """
        :param model: the model to convert
        :param output_dir: the path to a directory in which the converted model shall be stored. Note that multiple
            files will be written. If the directory does not exist, it will be created.
        """
        log.info("Checking model features")
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=self.TASK)
        onnx_config = model_onnx_config(model.config)
        self._convert(model, onnx_config, tokenizer, output_dir)

    def convert_santacoder_dual_models(self, model: PreTrainedModel, tokenizer, output_dir: Path):
        """
        Converts a model using the bigcode/santacoder architecture to two models (one without past
        and one with past), in order to make use of caching during inference.
        Each model with be stored in a subdirectory.

        :param model: the santacoder-based model
        :param tokenizer: the tokenizer
        :param output_dir: the
        :return:
        """
        # create the first model, which must output the values which can subsequently be used as past values
        # by the second model, using the modified configuration class which allows us to configure this
        log.info("Creating first model (without past) ...")
        onnx_config_without_past = SantaCoderOnnxConfig(model.config, task=self.TASK, use_past_outputs=True)
        self._convert(model, onnx_config_without_past, tokenizer, output_dir / self.SUBMODEL_DIR_WITHOUT_PAST)
        # create the second model with past
        log.info("Creating second model (with past) ...")
        onnx_config_with_past = SantaCoderOnnxConfig(model.config, task=self.TASK, use_past=True)
        self._convert(model, onnx_config_with_past, tokenizer, output_dir / self.SUBMODEL_DIR_WITH_PAST)

    @staticmethod
    def _quantize(onnx_src_dir: Path, onnx_target_dir: Path, qconfig):
        """
        :param onnx_src_dir: source directory containing one or more .onnx models to be quantized
        :param onnx_target_dir: target directory in which to save the quantized models to be generated
        :param qconfig: the quantization configuration
        """
        models = list(onnx_src_dir.glob("*.onnx"))
        if len(models) == 0:
            raise ValueError(f"No .onnx files found in {onnx_src_dir}")
        for onnx_file_path in models:
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
