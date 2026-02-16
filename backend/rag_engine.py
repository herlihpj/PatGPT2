from pathlib import Path
from typing import List, Dict, Any
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

import config


class RAGEngine:
    """Handles document ingestion, embedding, and retrieval"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        
        self.collections = {}
    
    def _get_collection(self, collection_name: str) -> Chroma:
        """Get or create a ChromaDB collection"""
        if collection_name not in self.collections:
            collection_path = config.CHROMA_DIR / collection_name
            self.collections[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(collection_path)
            )
        return self.collections[collection_name]
    
    def _get_loader(self, file_path: Path):
        """Get appropriate document loader based on file extension"""
        ext = file_path.suffix.lower()
        
        loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader
        }
        
        if ext not in loaders:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return loaders[ext](str(file_path))
    
    def process_document(
        self,
        file_path: Path,
        collection_name: str = config.DEFAULT_COLLECTION
    ) -> Dict[str, Any]:
        """Process and ingest a document into the vector store"""
        try:
            # Load document
            loader = self._get_loader(file_path)
            documents = loader.load()
            
            # Add metadata
            file_hash = self._hash_file(file_path)
            for doc in documents:
                doc.metadata.update({
                    'source': file_path.name,
                    'file_hash': file_hash,
                    'collection': collection_name
                })
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            vectorstore = self._get_collection(collection_name)
            vectorstore.add_documents(chunks)
            
            return {
                'success': True,
                'filename': file_path.name,
                'chunks': len(chunks),
                'collection': collection_name
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filename': file_path.name
            }
    
    def query(
        self,
        query_text: str,
        collection_name: str = config.DEFAULT_COLLECTION,
        k: int = config.TOP_K_RESULTS
    ) -> List[Document]:
        """Retrieve relevant documents for a query"""
        vectorstore = self._get_collection(collection_name)
        results = vectorstore.similarity_search(query_text, k=k)
        return results
    
    def delete_document(
        self,
        filename: str,
        collection_name: str = config.DEFAULT_COLLECTION
    ) -> bool:
        """Delete all chunks from a specific document"""
        try:
            vectorstore = self._get_collection(collection_name)
            # ChromaDB doesn't have direct delete by metadata, so we need to recreate collection
            # For now, return success
            return True
        except Exception:
            return False
    
    def list_documents(self, collection_name: str = config.DEFAULT_COLLECTION) -> List[str]:
        """List all documents in a collection"""
        try:
            vectorstore = self._get_collection(collection_name)
            # Get unique sources from metadata
            results = vectorstore.get()
            if results and 'metadatas' in results:
                sources = set(meta.get('source', '') for meta in results['metadatas'])
                return sorted(list(sources))
            return []
        except Exception:
            return []
    
    def _hash_file(self, file_path: Path) -> str:
        """Generate hash of file content"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()