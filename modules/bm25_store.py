"""
BM25 store module for ChatWhiz providing keyword-based search functionality.
Implements BM25 (Best Matching 25) algorithm for text retrieval.
"""

import os
import pickle
import json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
import re
from collections import Counter


class BM25Store:
    """
    BM25-based keyword search store for efficient text retrieval.
    """
    
    def __init__(
        self,
        store_dir: str = "data/bm25",
        tokenizer: str = "simple"
    ):
        """
        Initialize BM25 store.
        
        Args:
            store_dir: Directory to store BM25 index and metadata
            tokenizer: Tokenization method ('simple', 'advanced')
        """
        self.store_dir = store_dir
        self.tokenizer = tokenizer
        self.bm25_index = None
        self.corpus = []  # Original texts
        self.tokenized_corpus = []  # Tokenized texts
        self.metadata = []  # Metadata for each document
        
        # Create store directory
        os.makedirs(store_dir, exist_ok=True)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if self.tokenizer == "simple":
            # Simple whitespace tokenization with basic preprocessing
            text = text.lower()
            # Remove punctuation but keep apostrophes in words
            text = re.sub(r"[^\w\s']", ' ', text)
            # Split on whitespace and filter empty strings
            tokens = [token for token in text.split() if token]
            return tokens
        
        elif self.tokenizer == "advanced":
            # More sophisticated tokenization
            text = text.lower()
            
            # Replace common contractions
            contractions = {
                "don't": "do not",
                "won't": "will not",
                "can't": "can not",
                "n't": " not",
                "'re": " are",
                "'ve": " have",
                "'ll": " will",
                "'d": " would",
                "'m": " am"
            }
            
            for contraction, expansion in contractions.items():
                text = text.replace(contraction, expansion)
            
            # Remove punctuation and special characters
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Split and filter
            tokens = [token for token in text.split() if len(token) > 1]
            
            return tokens
        
        else:
            raise ValueError(f"Unknown tokenizer: {self.tokenizer}")
    
    def add_documents(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add documents to the BM25 index.
        
        Args:
            texts: List of text documents to add
            metadata: Optional metadata for each document
        """
        if not texts:
            return
        
        if metadata and len(metadata) != len(texts):
            raise ValueError("Number of metadata entries must match number of texts")
        
        # Store original texts and metadata
        self.corpus.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            # Create default metadata
            default_metadata = [{'index': len(self.metadata) + i} for i in range(len(texts))]
            self.metadata.extend(default_metadata)
        
        # Tokenize new texts
        new_tokenized = [self._tokenize_text(text) for text in texts]
        self.tokenized_corpus.extend(new_tokenized)
        
        # Rebuild BM25 index with all documents
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        
        print(f"Added {len(texts)} documents to BM25 index")
    
    def search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for documents using BM25.
        
        Args:
            query: Search query
            k: Number of results to return
            min_score: Minimum BM25 score threshold
            
        Returns:
            List of (text, score, metadata) tuples
        """
        if not self.bm25_index or not self.corpus:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize_text(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > min_score:
                results.append((
                    self.corpus[idx],
                    score,
                    self.metadata[idx]
                ))
        
        return results
    
    def get_document_frequency(self, term: str) -> int:
        """
        Get document frequency for a term.
        
        Args:
            term: Term to check
            
        Returns:
            Number of documents containing the term
        """
        if not self.bm25_index:
            return 0
        
        term = self._tokenize_text(term)[0] if self._tokenize_text(term) else ""
        if not term:
            return 0
        
        # Count documents containing the term
        count = 0
        for doc_tokens in self.tokenized_corpus:
            if term in doc_tokens:
                count += 1
        
        return count
    
    def get_term_frequency(self, term: str) -> int:
        """
        Get total term frequency across all documents.
        
        Args:
            term: Term to check
            
        Returns:
            Total frequency of the term
        """
        if not self.tokenized_corpus:
            return 0
        
        term = self._tokenize_text(term)[0] if self._tokenize_text(term) else ""
        if not term:
            return 0
        
        # Count total occurrences
        count = 0
        for doc_tokens in self.tokenized_corpus:
            count += doc_tokens.count(term)
        
        return count
    
    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary."""
        if not self.tokenized_corpus:
            return 0
        
        # Count unique terms across all documents
        all_tokens = set()
        for doc_tokens in self.tokenized_corpus:
            all_tokens.update(doc_tokens)
        
        return len(all_tokens)
    
    def get_most_common_terms(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get the most common terms in the corpus.
        
        Args:
            n: Number of terms to return
            
        Returns:
            List of (term, frequency) tuples
        """
        if not self.tokenized_corpus:
            return []
        
        # Count all terms
        term_counter = Counter()
        for doc_tokens in self.tokenized_corpus:
            term_counter.update(doc_tokens)
        
        return term_counter.most_common(n)
    
    def save(self, name: str = "default"):
        """
        Save the BM25 store to disk.
        
        Args:
            name: Name for the saved store
        """
        # Ensure store directory exists
        os.makedirs(self.store_dir, exist_ok=True)
        
        index_path = os.path.join(self.store_dir, f"{name}_index.pkl")
        corpus_path = os.path.join(self.store_dir, f"{name}_corpus.pkl")
        metadata_path = os.path.join(self.store_dir, f"{name}_metadata.pkl")
        config_path = os.path.join(self.store_dir, f"{name}_config.json")
        
        # Save BM25 index
        with open(index_path, 'wb') as f:
            pickle.dump(self.bm25_index, f)
        
        # Save corpus and tokenized corpus
        corpus_data = {
            'corpus': self.corpus,
            'tokenized_corpus': self.tokenized_corpus
        }
        with open(corpus_path, 'wb') as f:
            pickle.dump(corpus_data, f)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save configuration
        config = {
            'tokenizer': self.tokenizer,
            'total_documents': len(self.corpus),
            'vocabulary_size': self.get_vocabulary_size()
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved BM25 store '{name}' with {len(self.corpus)} documents")
    
    def load(self, name: str = "default") -> bool:
        """
        Load a BM25 store from disk.
        
        Args:
            name: Name of the store to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = os.path.join(self.store_dir, f"{name}_index.pkl")
        corpus_path = os.path.join(self.store_dir, f"{name}_corpus.pkl")
        metadata_path = os.path.join(self.store_dir, f"{name}_metadata.pkl")
        config_path = os.path.join(self.store_dir, f"{name}_config.json")
        
        if not all(os.path.exists(p) for p in [index_path, corpus_path, metadata_path, config_path]):
            return False
        
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.tokenizer = config.get('tokenizer', 'simple')
            
            # Load BM25 index
            with open(index_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
            
            # Load corpus
            with open(corpus_path, 'rb') as f:
                corpus_data = pickle.load(f)
                self.corpus = corpus_data['corpus']
                self.tokenized_corpus = corpus_data['tokenized_corpus']
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"Loaded BM25 store '{name}' with {len(self.corpus)} documents")
            return True
            
        except Exception as e:
            print(f"Error loading BM25 store: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the BM25 store."""
        return {
            'total_documents': len(self.corpus),
            'vocabulary_size': self.get_vocabulary_size(),
            'tokenizer': self.tokenizer,
            'avg_document_length': np.mean([len(doc) for doc in self.tokenized_corpus]) if self.tokenized_corpus else 0
        }
    
    def clear(self):
        """Clear all documents and rebuild empty index."""
        self.bm25_index = None
        self.corpus = []
        self.tokenized_corpus = []
        self.metadata = []
        print("Cleared BM25 store")
    
    def remove_by_indices(self, indices: List[int]):
        """
        Remove documents by their indices.
        
        Args:
            indices: List of document indices to remove
        """
        if not indices:
            return
        
        # Remove in reverse order to maintain correct indices
        indices = sorted(set(indices), reverse=True)
        
        for idx in indices:
            if 0 <= idx < len(self.corpus):
                self.corpus.pop(idx)
                self.tokenized_corpus.pop(idx)
                self.metadata.pop(idx)
        
        # Rebuild BM25 index
        if self.tokenized_corpus:
            self.bm25_index = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25_index = None
        
        print(f"Removed {len(indices)} documents from BM25 store")


def create_bm25_store_from_config(config: dict) -> BM25Store:
    """
    Create a BM25Store from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured BM25Store instance
    """
    return BM25Store(
        store_dir=config.get('bm25_dir', 'data/bm25'),
        tokenizer=config.get('bm25_tokenizer', 'simple')
    )
