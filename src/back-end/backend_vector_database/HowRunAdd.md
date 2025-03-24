```bash
   cd src/back-end
   python backend_vector_database/test_qdrant_connection.py
```


```bash
   cd src/back-end
   python backend_vector_database/create_vector_database.py --excel backend_vector_database/dataset/LegalRAG.xlsx
```

---
How check: https://e5180f0c-01ea-46a0-95c9-208559c12cef.europe-west3-0.gcp.cloud.qdrant.io:6333/dashboard#/collections

```
B 1 ± python backend_vector_database/create_vector_database.py --excel backend_vector_database/dataset/processed_text_v4.xlsx 
Loading data from backend_vector_database/dataset/processed_text_v4.xlsx...
Creating document objects with metadata...
Processing 2897 documents...
Initializing embeddings with model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
Creating vector database in Qdrant collection: DevOiMinhDiDauThe_RAG
Successfully added 2897 documents to Qdrant
1d [ubuntu@mgc-dev-3090-01:~/cuong_dn/ChatbotQA_DevGroup_Product_1_AITeam_CodeMely/src/back-end]└4 [.venv] main(+52/-43)* ± 
```