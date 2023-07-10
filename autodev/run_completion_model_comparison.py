"""
Compares several completion models by applying them to the example tasks in the data/completion-tasks directory,
creating a report in HTML format that contains all the results for inspection
"""

import logging
import sys

from autodev.autocomplete.completion_model_comparison import CompletionTaskModelComparison
from autodev.autocomplete.model import SantaCoderModelFactory

log = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout,
        level=logging.INFO)
    log.info("Starting")
    model_factory = SantaCoderModelFactory()

    general_models = ["bigcode/santacoder"]
    ruby_models = general_models + [
        "models/checkpoints/ruby/checkpoint-6000",
        "models/checkpoints/ruby/checkpoint-500"
    ]
    c_sharp_models = general_models + [
        "models/checkpoints/c-sharp/checkpoint-1000",
        "models/checkpoints/c-sharp/checkpoint-2000",
        "models/checkpoints/c-sharp/checkpoint-3000",
        "models/checkpoints/c-sharp/checkpoint-4000"
    ]
    rust_models = general_models + [
        "models/checkpoints/rust/checkpoint-30000"
    ]

    #CompletionTaskModelComparison("ruby", model_factory, model_paths=ruby_models).run()
    #CompletionTaskModelComparison("c-sharp", model_factory, model_paths=c_sharp_models).run()
    CompletionTaskModelComparison("rust", model_factory, model_paths=rust_models).run()
    #CompletionTaskModelComparison("scala", model_factory, model_paths=general_models).run()

    log.info("Done")
