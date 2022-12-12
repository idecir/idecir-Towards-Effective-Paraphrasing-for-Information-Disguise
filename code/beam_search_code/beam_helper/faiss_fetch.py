import os
from typing import Dict, List

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever


class FaissRetriever:
    """
    Initializes FAISS (https://ai.facebook.com/tools/faiss/) for retrieval.

    Attributes:
    faiss_file_name: path to the faiss index
    sql_lite_db_name: path to the actual contents of the vectors in the index
    """

    def __init__(self, exp_name: str):
        self.EXP_NAME: str = exp_name
        self.faiss_file_name: str = os.path.join(
            "../faiss_indexes", f"{self.EXP_NAME}.faiss")
        self.sql_lite_db_name: str = os.path.join(
            "../sql_lite_dbs", f"{self.EXP_NAME}_sql.db")

        PATH_TO_FAISS_INDEX: str = self.faiss_file_name
        self.document_store = FAISSDocumentStore(
            faiss_index_path=PATH_TO_FAISS_INDEX)

        self.retriever = DensePassageRetriever(
            document_store=self.document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
        )

    def fetch_top_k_docs(self, query_txt: str, k: int) -> List[Dict]:
        """Fetches rows from a Smalltable.

        Retrieves rows pertaining to the given keys from the Table instance
        represented by table_handle.

        Args:
            query_txt: string whose nearest matching documents needed to be fetched
            k: number of top documents to be fetched

        Returns:
        A list of relevant documents (dictionaries) with each document of the format:
            {
                "content": <Content of the document>
                "score": Relevance score of the document to the query,
                "post_id": POST ID from which the document was derived
            }
        """

        top_results = self.retriever.retrieve(
            query_txt, top_k=k, scale_score=True)
        top_results = [self.purify_doc(x.__dict__) for x in top_results]
        return top_results

    def fetch_results(self, q_text: str, post_id: str, k_val: int) -> Dict:
        """Fetches rows from a Smalltable.

        Retrieves rows pertaining to the given keys from the Table instance
        represented by table_handle.

        Args:
            q_txt: string whose nearest matching documents needed to be fetched
            post_id: POST ID of the source document from which the query was derived from
            k_val: number of top documents to be fetched

        Returns:
        Dictionary having the list of relevant documents and the best rank obtained by the source document
        """
        top_res = self.fetch_top_k_docs(q_text, k_val)
        best_doc_rank = 10**9
        for idx, doc in enumerate(top_res):
            if doc['post_id'] == post_id:
                best_doc_rank = idx + 1
                break
        # to consume less storage, return only the top 3 documents (while all
        # `k_val` were considered during ranking)
        return {"best_doc_rank": best_doc_rank, 'top_results': top_res[:3]}

    @staticmethod
    def purify_doc(doc):
        del doc['content_type']
        del doc['embedding']
        del doc['id']
        doc['post_id']: str = doc['meta']['post_id']
        del doc['meta']
        return doc
