from pathlib import Path

from autodev.util import logging
from autodev.autocomplete.model import SantaCoderModelFactory
from autodev.autocomplete.onnx_conversion import ONNXConversion

log = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.configure()
    log.info("Starting")

    # define the model to be converted
    model_factory = SantaCoderModelFactory()
    model_path = "bigcode/santacoder"

    # load model
    log.info("Loading model")
    model = model_factory.create_model(model_path)
    tokenizer = model_factory.create_tokenizer()

    # apply conversion
    ONNXConversion().convert(model, tokenizer, Path("models/santacoder_onnx"))
    #ONNXConversion().convert_santacoder_dual_models(model, tokenizer, Path("models/santacoder_onnx_cached"))

    log.info("Done")
