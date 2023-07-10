# AutoDev: LLM-Based Coding Assistance Functions

This is the AutoDev Python project for LLM-based coding assistance functions.
It provides:

* auto-completion models (that can suggest completions based on the current editing context)
  * fine-tuning of completion models to teach them new languages (or to teach them about your libraries, your code style, etc.)
  * quantitative & qualitative evaluation
  * optimization of models for inference (including quantization)

* code-based assistance functions, where an instruction-following model is given a task based on an existing code snippet (e.g. reviewing code, adding comments or input checks, explaining code, etc.)
* an inference service, which access to the above functions
* question answering on document databases (including source code documents)

## Environment

Use conda to set up your virtual environment:

    conda env create -f environment.yml

## Packages

The `autodev` package provides the following Python sub-packages and modules:
* `autocomplete` contains methods and abstractions for **auto-completion models**
  * `model` provides factories for the creation of (low-level) completion models
  * `completion_task` provides an abstraction for auto-completion tasks
  * `completion_model` contains the high-level abstraction for models that can solve such completion tasks
  * package `finetuning` is concerned with the fine-tuning of completion models (partly specific to bigcode/santacoder; see TODOs)
  * `completion_model_comparison`, `completion_report` and `perplexity_evaluation` are concerned with the qualitative and quantitative evaluation of completion models
  * `onnx_conversion` assists in optimizing/quantizing models for inference using ONNX runtime
* `llm` provides **abstractions for large language models** (LLMs).
  * The high-level abstraction, which provides streaming-based queries, is given by the `StreamingLLM` class.
  * Instances of `StreamingLLM` can be created via specializations of the `LLMFactory` class.
  * `LLMType` constitutes a convenient enumeration of the model types considered in concrete factory implementations.
* `code_functions` contains **code snippet-based assistant functions** that take code as input and return text/code as output, supporting editor-based
  actions where the user selects text in the editor and then uses the context menu to invoke an assistant function.
* `service` implements a **Flask-based service** that provides access to the aforementioned code functions via an HTTP service.
* `stream_formatting` is concerned with the on-the-fly HTML formatting of streamed service responses
* package `qa` is concerned with **question answering**
  * `document_db` provides abstractions for document databases
  * `qa_use_case` implements the full question answering use case based on a (static) document database using `langchain`
  * the additional further modules `splitting`, `indexing`, `embedding` are currently unused but could prove useful when extending the question answering use case to use fewer non-standard components that are directly provided by the `langchain` library
* `logging` facilitates logging configuration (as a drop-in replacement for Python's `logging` module)

## Runnable Scripts and Notebooks

The root folder contains a number of runnable scripts and notebooks.

In the main function of runnable scripts, you will typically find several job configurations (one per line), only one of which is not commented out.

### Auto-Completion Models

* `run_completion_model_finetuning.py` performs fine-tuning of `bigcode/santacoder` to teach it a new language, using data from `the-stack-dedup`.
* `run_completion_model_comparison.py` uses several completion models to solve completion tasks stores in `data/completion-tasks`, storing the results in an HTML report.
* `run_completion_model_perplexity_evaluation.py` performs a quantitative performance evaluation of completion models by computing the perplexity for previously unseed source code snippets.
* `run_onnx_conversion.py` applies ONNX version to completion models.
* `run_service.py` starts the HTTP service for remote access to coding assistance functions.
* Notebook `benchmark_completion_models.ipynb` contains an inference runtime evaluation of various quantized/optimized instances of the same model (CPU and GPU).

### Code Snippet-Based Assistance Functions

Notebook `apply_code_functions.ipynb` can be used to apply code functions to the code snippets in `data/code_snippets` using different LLMs.

### Question Answering

`run_qa_fireface_manual.py` and `run_qa_sensai.py` implement question answering use cases based on the manual of an audio interface and the [sensAI](http://github.com/jambit/sensAI) library source code respectively.

### Inference Service

`run_service.py` starts the inference service, which serves as a backend for the IntelliJ plugin.