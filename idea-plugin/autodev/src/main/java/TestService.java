import de.appliedai.autodev.ServiceClient;

import java.io.IOException;

public class TestService {
    public static void main(String[] args) throws IOException, InterruptedException {
        ServiceClient client = new ServiceClient();
        String code = "\tclass UseCase:\n" +
                "\t\tDB_ROOT_DIRECTORY = \"db\"\n" +
                "\n" +
                "\t\tdef __init__(self, llm_type: LLMType, doc_db: DocumentDatabase, splitter: TextSplitter, queries: List[str]):\n" +
                "\t\t\tself.llm_type = llm_type\n" +
                "\t\t\tself.doc_db = doc_db\n" +
                "\t\t\tself.queries = queries\n" +
                "\t\t\tself.splitter = splitter\n" +
                "\t\t\tself.db = self.get_or_create_db(load=True)\n" +
                "\t\t\tlog.info(f\"Creating model {llm_type}\")\n" +
                "\t\t\tllm = llm_type.create_llm()\n" +
                "\t\t\tself.qa = RetrievalQA.from_chain_type(llm=llm, chain_type=\"stuff\", retriever=self.db.as_retriever())\n" +
                "\n" +
                "\t\tdef _db_directory(self) -> str:\n" +
                "\t\t\treturn f\"{self.DB_ROOT_DIRECTORY}/{self.doc_db.name}\"\n" +
                "\n" +
                "\t\tdef get_or_create_db(self, load=True) -> Chroma:\n" +
                "\t\t\tembedding_function = OpenAIEmbeddings()\n" +
                "\t\t\tif load and os.path.exists(os.path.join(self._db_directory(), \"chroma-embeddings.parquet\")):\n" +
                "\t\t\t\tdb = Chroma(embedding_function=embedding_function, persist_directory=self._db_directory())\n" +
                "\t\t\telse:\n" +
                "\t\t\t\ttexts = self.splitter.split_documents(self.doc_db.documents)\n" +
                "\t\t\t\tlog.info(f\"Documents were split into {len(texts)} sub-documents\")\n" +
                "\n" +
                "\t\t\t\tdb = Chroma.from_documents(texts, embedding_function, persist_directory=self._db_directory())\n" +
                "\t\t\t\tdb.persist()\n" +
                "\t\t\treturn db\n" +
                "\n" +
                "\t\tdef query(self, q):\n" +
                "\t\t\tprint(f\"\\n{q}\")\n" +
                "\t\t\tanswer = self.qa.run(q)\n" +
                "\t\t\tprint(answer)\n" +
                "\n" +
                "\t\tdef run_example_queries(self):\n" +
                "\t\t\tfor q in self.queries:\n" +
                "\t\t\t\tself.query(q)\n";
        String result = client.addComments(code);
        System.out.println(result);
    }
}
