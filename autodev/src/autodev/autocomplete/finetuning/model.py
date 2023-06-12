import logging
import os
from typing import Union

from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedModel

log = logging.getLogger(__name__)


def model_id_to_fn(model_id: str):
    return model_id.replace("/", "--")


def model_id_from_fn(model_fn: str):
    return model_fn.replace("--", "/")


def get_model(model_path: str, base_model_id: str) -> Union[PreTrainedModel, PeftModel]:
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)
        log.info(f"Loading PEFT model from {model_path}")
        return PeftModel.from_pretrained(base_model, model_path)
    else:
        return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
