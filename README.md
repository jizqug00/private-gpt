# 🔒 PrivateGPT 📑

My PrivateGPT, modified from the repository [zylon-ai/private-gpt](https://github.com/zylon-ai/private-gpt) is a production-ready AI project that allows you to ask questions about multimedia (Documents, Images, Audios, URLs) using the power of Large Language Models (LLMs), even in scenarios without an Internet connection. 100% private, no data leaves your
execution environment at any point.

The project provides an API offering all the primitives required to build private, context-aware AI applications.
It follows and extends the [OpenAI API standard](https://openai.com/blog/openai-api),
and supports both normal and streaming responses.

The API is divided into two logical blocks:

**High-level API**, which abstracts all the complexity of a RAG (Retrieval Augmented Generation)
pipeline implementation:
- Ingestion of documents: internally managing document parsing,
splitting, metadata extraction, embedding generation and storage.
- Chat & Completions using context from ingested documents:
abstracting the retrieval of context, the prompt engineering and the response generation.

**Low-level API**, which allows advanced users to implement their own complex pipelines:
- Embeddings generation: based on a piece of text.
- Contextual chunks retrieval: given a query, returns the most relevant chunks of text from the ingested documents.

In addition to this, a working [Gradio UI](https://www.gradio.app/)
client is provided to test the API, together with a set of useful tools such as bulk model
download script, ingestion script, documents folder watch, etc.

## 📄 Documentation
Full documentation on installation, dependencies, configuration, running the server, deployment options,
ingesting local documents, API details and UI features can be found here: https://docs.privategpt.dev/

## 💡 My Installation
Steps I followed in my Windows installation, before the procedure [Ollama](https://ollama.com/), Cuda and the libraries pillow, googletrans and transformaers need to be installed in the machine, these last libraries are because of the need to execute this external python script vlm.py to proccess the images with the VLM moondream2 avoiding problems with dependencies:

### 1. Clone the PrivateGPT Repository
Clone the repository and navigate to it:

    - git clone https://github.com/zylon-ai/private-gpt
    - cd private-gpt

### 2. Install Python 3.11
If you do not have Python 3.11 installed, install it using a Python version manager like pyenv. Earlier Python versions are not supported.

#### macOS/Linux
Install and set Python 3.11 using pyenv:

    - pyenv install 3.11
    - pyenv local 3.11

#### Windows
Install and set Python 3.11 using pyenv-win:

    - pyenv install 3.11
    - pyenv local 3.11

### 3. Install Poetry
Install Poetry for dependency management: Follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installing-with-the-official-installer) to install it.

### 4. Install make
To run various scripts, you need to install make. Follow the instructions for your operating system:

#### macOS
(Using Homebrew):

    - brew install make

#### Windows
(Using Chocolatey):

    - choco install make

### 5. Local, Ollama-powered setup
Now, on a different terminal, start Ollama service (it will start a local inference server, serving both the LLM and the Embeddings):

    - ollama serve

Once done, you can install PrivateGPT with the following command:

    - poetry install --extras "ui llms-ollama embeddings-ollama vector-stores-qdrant"

And finally run PrivateGPT with the Ollama configuration:

#### Windows
    - $env:PGPT_PROFILES="ollama"
    - make run

#### macOS/Linux
    - set PGPT_PROFILES=ollama
    - make run

After these commands private-gpt will be running in localhost:8001. The first time it is executed the models will be downloaded when asked for them (upload files, upload audios, upload images and chatting).

## 🧩 Architecture
Conceptually, PrivateGPT is an API that wraps a RAG pipeline and exposes its
primitives.
* The API is built using [FastAPI](https://fastapi.tiangolo.com/) and follows
  [OpenAI's API scheme](https://platform.openai.com/docs/api-reference).
* The RAG pipeline is based on [LlamaIndex](https://www.llamaindex.ai/).

The design of PrivateGPT allows to easily extend and adapt both the API and the
RAG implementation. Some key architectural decisions are:
* Dependency Injection, decoupling the different components and layers.
* Usage of LlamaIndex abstractions such as `LLM`, `BaseEmbedding` or `VectorStore`,
  making it immediate to change the actual implementations of those abstractions.
* Simplicity, adding as few layers and new abstractions as possible.
* Ready to use, providing a full implementation of the API and RAG
  pipeline.

Main building blocks:
* APIs are defined in `private_gpt:server:<api>`. Each package contains an
  `<api>_router.py` (FastAPI layer) and an `<api>_service.py` (the
  service implementation). Each *Service* uses LlamaIndex base abstractions instead
  of specific implementations,
  decoupling the actual implementation from its usage.
* Components are placed in
  `private_gpt:components:<component>`. Each *Component* is in charge of providing
  actual implementations to the base abstractions used in the Services - for example
  `LLMComponent` is in charge of providing an actual implementation of an `LLM`
  (for example `LlamaCPP` or `OpenAI`).


