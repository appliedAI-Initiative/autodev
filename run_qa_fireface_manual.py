import logging
import os
import sys

from langchain.text_splitter import CharacterTextSplitter

from autodev.document_qa import UseCase, SingleTextFileDocumentDatabase
from autodev.llm import LLMType

log = logging.getLogger(__name__)


class DocumentDatabaseFirefaceManual(SingleTextFileDocumentDatabase):
    def __init__(self):
        super().__init__("fireface", os.path.join('data', 'fface_uc_e.txt'))


class UseCaseFirefaceManual(UseCase):
    def __init__(self, llm_type: LLMType):
        queries = [
            "What is the impedance of the instrument input?",
            "What is the purpose of the matrix mixer?",
            "How many mic inputs does the device have?",
            "Can the device be powered via USB alone?",
            "What is the minimum round trip latency?",
            "How can I invert stereo for an output channel?",
            "How can I change the headphone output volume on the device itself?",
            "What is a submix and how many submixes can be defined?",
        ]
        super().__init__(llm_type=llm_type, doc_db=DocumentDatabaseFirefaceManual(),
            splitter=CharacterTextSplitter(chunk_size=llm_type.chunk_size(), chunk_overlap=0),
            queries=queries)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s', stream=sys.stdout, level=logging.INFO)

    use_case = UseCaseFirefaceManual(LLMType.OPENAI_DAVINCI3)
    #use_case = UseCaseFirefaceManual(LLMType.OPENAI_CHAT_GPT4)
    #use_case = UseCaseFirefaceManual(LLMType.HUGGINGFACE_GPT2)

    use_case.run_example_queries()

