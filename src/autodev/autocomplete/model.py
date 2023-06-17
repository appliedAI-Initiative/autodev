import logging
import os
from typing import Union

from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, AutoTokenizer

log = logging.getLogger(__name__)


def model_id_to_fn(model_id: str):
    return model_id.replace("/", "--")


def model_id_from_fn(model_fn: str):
    return model_fn.replace("--", "/")


class ModelFactory:
    def __init__(self, base_model_id: str):
        self.base_model_id = base_model_id

    def create_tokenizer(self) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)

    def create_model(self, model_path) -> Union[PreTrainedModel, PeftModel]:
        """
        :param model_path: a path to a directory containing the model/checkpoint to load or
            a path/identifier known to the transformers library
        :return: the model
        """
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_id, trust_remote_code=True)
            log.info(f"Loading PEFT model from {model_path}")
            return PeftModel.from_pretrained(base_model, model_path)
        else:
            return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)


class SantaCoderModelFactory(ModelFactory):
    def __init__(self):
        super().__init__(base_model_id="bigcode/santacoder")