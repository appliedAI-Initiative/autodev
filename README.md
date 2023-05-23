# AutoDev: LLM-Based Coding Assistance Functions

This repository contains two projects:

* The **AutoDev Python library** providing the core functionality (`./`)
* A Java project implementing the **AutoDev IntellIJ IDEA plugin** that provides access to the coding assistance functions within an IDE (`./idea-plugin/autodev-plugin`).

## AutoDev Library

### Environment

Use [poetry](https://python-poetry.org) to set up your virtual environment using a Python 3.10 base.

### Packages

The `autodev` package provides the following Python modules:
* `llm` provides **abstractions for large language models** (LLMs).
  * The high-level abstraction, which provides streaming-based queries, is given by the `StreamingLLM` class.
  * Instances of `StreamingLLM` can be created via specializations of the `LLMFactory` class.
  * `LLMType` constitutes a convenient enumeration of the model types considered in concrete factory implementations.
* `code_functions` contains **code assistant functions** that take code as input and return text/code as output, supporting editor-based
  actions where the user selects text in the editor and then uses the context menu to invoke an assistant function.
* `service` implements a **Flask-based service** that provides access to the aforementioned code functions via an HTTP service.
* `stream_formatting` is concerned with the on-the-fly HTML formatting of streamed responses
* `document_qa` provides simple **question answering** functionality (based on a static set of documents)
* further modules (`splitting`, `indexing`, `embedding`) that could prove useful when extending the question answering use case to use fewer non-standard components that are directly provided by the `langchain` library
* `logging` which facilitates logging configuration (as a drop-in replacement for Python's `logging` module)

### Runnable Scripts

The root folder contains runnable scripts:
* `run_qa_fireface_manual.py` and `run_qa_sensai.py` implement question answering use cases based on the manual of an audio interface and the [sensAI](http://github.com/jambit/sensAI) library source code respectively.
* `run_service.py` starts the HTTP service for remote access to coding assistance functions.

Notebooks:
* `apply_code_functions.ipynb` can be used to apply code functions to the code snippets in `data/code_snippets` using different LLMs.

## AutoDev IntellIJ IDEA Plugin

Open the folder `idea-plugin/autodev-plugin` as a project in IntellIJ.

IntelliJ should detect the gradle project and display the run configuration `Run Plugin`.
Running this configuration will start an IntellIJ instance with the plugin enabled.
By default, it will query the service at `localhost:5000`.

### Components

* The file `plugin.xml` defines the plugin components that are activated
* The package `de.appliedai.autodev.actions` contains editor actions (available in the editor context menu)
* Class `de.appliedai.autodev.ServiceClient` contains the service client implementation, in which the service URL is configured.
* Class `de.appliedai.autodev.AutoDevToolWindowManager` manages the creation of tool window components/tabs that are displayed in reaction to user queries.

