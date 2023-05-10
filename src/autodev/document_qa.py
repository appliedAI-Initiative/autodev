import logging
import os
from typing import List

from langchain.chains import RetrievalQA
from langchain.document_loaders import PythonLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import Chroma

from autodev.llm import LLMType

log = logging.getLogger(__name__)


class DocumentDatabase:
    def __init__(self, name: str, documents: List[Document]):
        self.name = name
        self.documents = documents


class SingleTextFileDocumentDatabase(DocumentDatabase):
    def __init__(self, name: str, textfile: str):
        super().__init__(name, TextLoader(textfile).load())


class PythonDocumentDatabase(DocumentDatabase):
    def __init__(self, name: str, src_root: str):
        documents = []
        for root, dirs, files in os.walk(src_root):
            for fn in files:
                fn: str
                if fn.endswith(".py"):
                    pypath = os.path.join(root, fn)
                    documents.extend(PythonLoader(pypath).load())
        super().__init__(name, documents)


class VectorDatabase:
    DB_ROOT_DIRECTORY = "vectordb"

    def __init__(self, name: str, doc_db: DocumentDatabase, splitter: TextSplitter, embedding_function: Embeddings,
            load=True):
        self.name = name
        self.embedding_function = embedding_function
        self.doc_db = doc_db
        self.splitter = splitter
        self.db = self._get_or_create_db(load=load)

    def _db_directory(self) -> str:
        return f"{self.DB_ROOT_DIRECTORY}/{self.name}"

    def _get_or_create_db(self, load=True) -> Chroma:
        if load and os.path.exists(os.path.join(self._db_directory(), "chroma-embeddings.parquet")):
            db = Chroma(embedding_function=self.embedding_function, persist_directory=self._db_directory())
        else:
            texts = self.splitter.split_documents(self.doc_db.documents)
            log.info(f"Documents were split into {len(texts)} sub-documents")

            db = Chroma.from_documents(texts, self.embedding_function, persist_directory=self._db_directory())
            db.persist()
        return db

    def retriever(self) -> BaseRetriever:
        return self.db.as_retriever()


class UseCase:
    """
    Represents a question answering use case
    """
    def __init__(self, llm_type: LLMType, doc_db: DocumentDatabase, splitter: TextSplitter, queries: List[str]):
        """
        :param llm_type: An LLMType object, which specifies the type of LLM model to use.
        :param doc_db: A DocumentDatabase object, which contains the text documents for querying.
        :param splitter: A TextSplitter object, which is used to split the documents into sub-documents.
        :param queries: A list of strings, representing example queries that can be executed.
        """
        self.llm_type = llm_type
        self.doc_db = doc_db
        self.queries = queries
        self.splitter = splitter
        self.vector_db = VectorDatabase(doc_db.name, doc_db, splitter, OpenAIEmbeddings())
        log.info(f"Creating model {llm_type}")
        llm = llm_type.create_llm()
        self.qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=self.vector_db.retriever())

    def query(self, q):
        print(f"\n{q}")
        answer = self.qa.run(q)
        print(answer)

    def run_example_queries(self):
        for q in self.queries:
            self.query(q)
