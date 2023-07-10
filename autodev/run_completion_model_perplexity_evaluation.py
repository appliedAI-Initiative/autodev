import logging
import sys

from autodev.autocomplete.model import SantaCoderModelFactory
from autodev.autocomplete.preplexity_evaluation import ModelPerplexityEvaluation

log = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout,
        level=logging.INFO)
    model_factory = SantaCoderModelFactory()
    ruby_models = [
        "models/checkpoints/ruby-lora16-fp32/checkpoint-3500",
        "models/checkpoints/ruby-lora64-fp32/checkpoint-3000",
        "models/checkpoints/ruby/checkpoint-3000",
        "models/checkpoints/ruby/checkpoint-500",
        "models/checkpoints/ruby/checkpoint-1000",
        "models/checkpoints/ruby/checkpoint-2000",
        "models/checkpoints/ruby/checkpoint-6000",
        "bigcode/santacoder"
    ]
    csharp_models = [
        "models/checkpoints/c-sharp/checkpoint-1000",
        "models/checkpoints/c-sharp/checkpoint-2000",
        "models/checkpoints/c-sharp/checkpoint-3000",
        "models/checkpoints/c-sharp/checkpoint-4000",
        "bigcode/santacoder"
    ]
    rust_models = [
        "models/checkpoints/rust/checkpoint-30000",
        "bigcode/santacoder"
    ]
    # ModelPerplexityEvaluation("ruby", model_factory, ruby_models).run()
    # ModelPerplexityEvaluation("c-sharp", model_factory, csharp_models).run()
    ModelPerplexityEvaluation("rust", model_factory, rust_models).run()
