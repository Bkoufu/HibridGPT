# HybridGPT

we offer a hybrid mode that combines the power of local LLMs and OpenAI models, allowing for versatile and comprehensive responses to your queries.
It is very fast based on a caching system . see below # How does it work?


Built with [LangChain](https://github.com/hwchase17/langchain), [GPT4All](https://github.com/nomic-ai/gpt4all), [LlamaCpp](https://github.com/ggerganov/llama.cpp), [Chroma](https://www.trychroma.com/) and [SentenceTransformers](https://www.sbert.net/).

<img width="902" alt="demo" src="https://github.com/Bkoufu/HibridGPT/blob/main/private.gif">

# Environment Setup
In order to set your environment up to run the code here, first install all requirements:

```shell
pip3 install -r requirements.txt
```

Then, download the LLM model and place it in a directory of your choice:
- LLM: default to [ggml-gpt4all-j-v1.3-groovy.bin](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin). If you prefer a different GPT4All-J compatible model, just download it and reference it in your `.env` file.

Rename `example.env` to `.env` and edit the variables appropriately.
```
MODEL_TYPE: supports LlamaCpp or GPT4All
PERSIST_DIRECTORY: is the folder you want your vectorstore in
MODEL_PATH: Path to your GPT4All or LlamaCpp supported LLM
MODEL_N_CTX: Maximum token limit for the LLM model
EMBEDDINGS_MODEL_NAME: SentenceTransformers embeddings model name (see https://www.sbert.net/docs/pretrained_models.html)
TARGET_SOURCE_CHUNKS: The amount of chunks (sources) that will be used to answer a question
```

Note: because of the way `langchain` loads the `SentenceTransformers` embeddings, the first time you run the script it will require internet connection to download the embeddings model itself.

## Test dataset
This repo uses a [state of the union transcript](https://github.com/imartinez/privateGPT/blob/main/source_documents/state_of_the_union.txt) as an example.

## Instructions for ingesting your own dataset

Put any and all your files into the `source_documents` directory

The supported extensions are:

   - `.csv`: CSV,
   - `.docx`: Word Document,
   - `.doc`: Word Document,
   - `.enex`: EverNote,
   - `.eml`: Email,
   - `.epub`: EPub,
   - `.html`: HTML File,
   - `.md`: Markdown,
   - `.msg`: Outlook Message,
   - `.odt`: Open Document Text,
   - `.pdf`: Portable Document Format (PDF),
   - `.pptx` : PowerPoint Document,
   - `.ppt` : PowerPoint Document,
   - `.txt`: Text file (UTF-8),
   - `.xlxs`: excel file ,

Run the following command to ingest all the data.

```shell
python ingest.py
```

Output should look like this:

```shell
Creating new vectorstore
Loading documents from source_documents
Loading new documents: 100%|██████████████████████| 1/1 [00:01<00:00,  1.73s/it]
Loaded 1 new documents from source_documents
Split into 90 chunks of text (max. 500 tokens each)
Creating embeddings. May take some minutes...
Using embedded DuckDB with persistence: data will be stored in: db
Ingestion complete! You can now run privateGPT.py to query your documents
```

It will create a `db` folder containing the local vectorstore. Will take 20-30 seconds per document, depending on the size of the document.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `db` folder.

Note: during the ingest process no data leaves your local environment. You could ingest without an internet connection, except for the first time you run the ingest script, when the embeddings model is downloaded.

## Ask questions to your documents, locally!
In order to ask a question, run a command like:

```shell
python HybridGPT.py
```

And wait for the script to require your input.

```plaintext
> Enter a query:
```

Hit enter. You'll need to wait 20-30 seconds (depending on your machine) while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again.

Note: you could turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.


# How does it work?
Selecting the right local models and the power of `LangChain` you can run the entire pipeline locally, without any data leaving your environment, and with reasonable performance.

- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `HuggingFaceEmbeddings` (`SentenceTransformers`). It then stores the result in a local vector database using `Chroma` vector store.
-  HybridGPT.py enables a hybrid mode, leveraging the power of both local LLM based on GPT4All-J or LlamaCpp and OpenAI models to understand questions and generate answers. Depending on the configuration, it either retrieves context for the answers from the local vector store using a similarity search, or it directly accesses the OpenAI model for responses. This versatile setup allows you to get the best of both worlds, harnessing the privacy of local computations and the advanced capabilities of OpenAI models.
- `GPT4All-J` wrapper was introduced in LangChain 0.0.162.
-  In our hybrid GPT system, we've introduced a class FaissClass to handle the efficient similarity search and indexing of dense vectors. This class is implemented using Faiss, a library developed by Facebook Research, known for its efficiency in handling large scale vector similarity search and clustering.
    Faiss allows for quick nearest neighbor search in high dimensional spaces, which is crucial in our application where each document or chunk of text is represented as a high-dimensional vector. Using Faiss, we can quickly retrieve the most relevant documents or chunks to a given query, providing the context necessary for the model to generate a coherent response.
-  Our hybrid GPT system features the SQLiteCache class for efficient caching of responses. This class is implemented using SQLite, a C library that provides a lightweight, disk-based database. SQLite allows the system to store data across sessions, which is particularly useful for caching responses to avoid unnecessary re-computation.

# System Requirements

## Python Version
To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler
If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

### For Windows 10/11
To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
   * Universal Windows Platform development
   * C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the `gcc` component.

##Mac and linux is not tested 

# Disclaimer
This is a prototype project designed to evaluate the viability of a hybrid, privacy-preserving solution for question-answering tasks, leveraging both local language models (LLMs) and vector embeddings. While the system can function both online and offline, it is not yet production-ready and is therefore not intended for use in production environments. The choice of models has been optimized primarily for privacy and can be tailored to enhance performance, including the use of different models and vector stores. Potential use cases of this project may be modified according to the specific needs and constraints of the user.

# Thanks to

https://github.com/imartinez/privateGPT
