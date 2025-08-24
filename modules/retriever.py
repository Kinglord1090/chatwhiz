"""
Retrieval system for ChatWhiz supporting semantic, BM25, and hybrid search modes.
"""

import os
import pickle
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi

try:
    from .embedder import InstructorEmbedder
    from .vector_store import FAISSVectorStore
    from .bm25_store import BM25Store, create_bm25_store_from_config
    from .loader import ChatMessage
except ImportError:
    # Fallback to absolute imports when running as standalone module
    from embedder import InstructorEmbedder
    from vector_store import FAISSVectorStore
    from bm25_store import BM25Store, create_bm25_store_from_config
    from loader import ChatMessage


class SearchResult:
    """Represents a search result with score and metadata."""
    
    def __init__(
        self,
        text: str,
        score: float,
        metadata: Dict[str, Any],
        search_type: str = "semantic"
    ):
        self.text = text
        self.score = score
        self.metadata = metadata
        self.search_type = search_type
    
    def __repr__(self):
        return f"SearchResult(score={self.score:.3f}, type={self.search_type})"


class ChatRetriever:
    """
    Multi-modal retrieval system supporting semantic, BM25, and hybrid search.
    """
    
    def __init__(
        self,
        embedder: InstructorEmbedder,
        vector_store: FAISSVectorStore,
        bm25_store: Optional[BM25Store] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the retriever.

        Args:
            embedder: Instructor embedder for semantic search
            vector_store: FAISS vector store
            bm25_store: BM25 store for keyword search (optional)
            config: Configuration dictionary
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config or {}
        
        # Initialize BM25 store
        if bm25_store is None:
            self.bm25_store = create_bm25_store_from_config(self.config)
        else:
            self.bm25_store = bm25_store
        
        # Legacy BM25 support (for backward compatibility)
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_metadata = []

        # Automatically load existing indexes on startup
        self._load_existing_indexes()

    def _load_existing_indexes(self):
        """Load existing vector store and BM25 indexes if they exist."""
        try:
            # Try to load vector store
            if self.vector_store.load("default"):
                print(f"Loaded existing vector store with {self.vector_store.get_stats()['total_vectors']} vectors")

            # Try to load new BM25 store
            if self.bm25_store.load("default"):
                print(f"Loaded existing BM25 store with {self.bm25_store.get_stats()['total_documents']} documents")
            # Fallback to legacy BM25 index
            elif self._load_bm25_index():
                print(f"Loaded legacy BM25 index with {len(self.bm25_corpus)} documents")

        except Exception as e:
            print(f"Note: Could not load existing indexes: {e}")
            # This is not an error - just means no indexes exist yet

    def index_messages_with_progress(self, messages: List[ChatMessage], rebuild: bool = False, batch_size: int = 100, progress_callback=None):
        """
        Index chat messages with progress reporting.

        Args:
            messages: List of chat messages to index
            rebuild: Whether to rebuild existing indices
            batch_size: Number of messages to process in each batch
            progress_callback: Callback function(current, total, stage) for progress updates
        """
        if not messages:
            print("No messages to index")
            return

        total_messages = len(messages)
        print(f"Indexing {total_messages} messages with granular progress...")

        # Index for semantic search
        if rebuild or self.vector_store.get_stats()['total_vectors'] == 0:
            print("Creating semantic embeddings...")

            if rebuild:
                self.vector_store.clear()

            # Process in smaller batches for more granular progress
            total_batches = (total_messages + batch_size - 1) // batch_size
            processed = 0

            for batch_idx in range(0, total_messages, batch_size):
                batch_end = min(batch_idx + batch_size, total_messages)
                batch_messages = messages[batch_idx:batch_end]
                batch_num = batch_idx // batch_size + 1

                # Report progress
                if progress_callback:
                    progress_callback(processed, total_messages, "Creating embeddings")

                # Extract texts and metadata for this batch
                texts = [f"{msg.sender}: {msg.text}" for msg in batch_messages]
                metadata = [self._get_message_metadata(msg) for msg in batch_messages]

                # Generate embeddings for this batch
                embeddings = self.embedder.encode(texts, use_cache=True, batch_size=min(16, len(texts)))

                # Add to vector store
                self.vector_store.add_embeddings(embeddings, texts, metadata)
                
                processed += len(batch_messages)
                
                # Report batch completion
                if progress_callback:
                    progress_callback(processed, total_messages, "Creating embeddings")

            # Save vector store
            print("Saving vector store...")
            self.vector_store.save("default")

        # Index for BM25 search
        if rebuild or self.bm25_store.get_stats()['total_documents'] == 0:
            print("Creating BM25 index...")
            
            if progress_callback:
                progress_callback(0, total_messages, "Building BM25 index")
            
            if rebuild:
                self.bm25_store.clear()
            
            texts = [f"{msg.sender}: {msg.text}" for msg in messages]
            metadata = [self._get_message_metadata(msg) for msg in messages]
            
            # Add to BM25 store
            self.bm25_store.add_documents(texts, metadata)
            self.bm25_store.save("default")
            
            # Also maintain legacy index
            self._create_bm25_index(texts, metadata)
            self._save_bm25_index()
            
            if progress_callback:
                progress_callback(total_messages, total_messages, "BM25 index complete")

        print("Indexing completed!")

    def index_messages(self, messages: List[ChatMessage], rebuild: bool = False, batch_size: int = 1000):
        """
        Index chat messages for both semantic and BM25 search with batch processing.

        Args:
            messages: List of chat messages to index
            rebuild: Whether to rebuild existing indices
            batch_size: Number of messages to process in each batch
        """
        if not messages:
            print("No messages to index")
            return

        print(f"Indexing {len(messages)} messages in batches of {batch_size}...")

        # Index for semantic search
        if rebuild or self.vector_store.get_stats()['total_vectors'] == 0:
            print("Creating semantic embeddings...")

            if rebuild:
                self.vector_store.clear()

            # Process in batches to avoid memory issues
            total_batches = (len(messages) + batch_size - 1) // batch_size

            for batch_idx in range(0, len(messages), batch_size):
                batch_end = min(batch_idx + batch_size, len(messages))
                batch_messages = messages[batch_idx:batch_end]
                batch_num = batch_idx // batch_size + 1

                print(f"  Processing batch {batch_num}/{total_batches}: {len(batch_messages)} messages")

                # Extract texts and metadata for this batch
                texts = [f"{msg.sender}: {msg.text}" for msg in batch_messages]
                metadata = [self._get_message_metadata(msg) for msg in batch_messages]

                # Generate embeddings for this batch
                embeddings = self.embedder.encode(texts, use_cache=True, batch_size=16)

                # Add to vector store using the correct method
                self.vector_store.add_embeddings(embeddings, texts, metadata)

                print(f"    Added {len(batch_messages)} vectors to index")

            # Save vector store
            print("Saving vector store...")
            self.vector_store.save("default")

        # Index for BM25 search using new BM25Store
        if rebuild or self.bm25_store.get_stats()['total_documents'] == 0:
            print("Creating BM25 index...")
            if rebuild:
                self.bm25_store.clear()
            
            texts = [f"{msg.sender}: {msg.text}" for msg in messages]
            metadata = [self._get_message_metadata(msg) for msg in messages]
            
            # Add to new BM25 store
            self.bm25_store.add_documents(texts, metadata)
            self.bm25_store.save("default")
            
            # Also maintain legacy index for backward compatibility
            self._create_bm25_index(texts, metadata)
            self._save_bm25_index()

        print("Indexing completed!")

    def _get_message_metadata(self, message: ChatMessage) -> Dict[str, Any]:
        """Get metadata for a single message."""
        return {
            'message_id': message.message_id,
            'sender': message.sender,
            'timestamp': message.timestamp.isoformat(),
            'message_count': 1,
            'senders': [message.sender]
        }

    def _create_bm25_index(self, texts: List[str], metadata: List[Dict[str, Any]]):
        """Create BM25 index from texts."""
        # Tokenize texts (simple whitespace tokenization)
        tokenized_corpus = [text.lower().split() for text in texts]
        
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = texts
        self.bm25_metadata = metadata
    
    def _save_bm25_index(self):
        """Save BM25 index to disk."""
        store_dir = self.config.get('processed_dir', 'data/processed')
        os.makedirs(store_dir, exist_ok=True)
        
        bm25_path = os.path.join(store_dir, "bm25_index.pkl")
        corpus_path = os.path.join(store_dir, "bm25_corpus.pkl")
        metadata_path = os.path.join(store_dir, "bm25_metadata.pkl")
        
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.bm25_index, f)
        
        with open(corpus_path, 'wb') as f:
            pickle.dump(self.bm25_corpus, f)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.bm25_metadata, f)
        
        print("BM25 index saved")
    
    def _load_bm25_index(self) -> bool:
        """Load BM25 index from disk."""
        store_dir = self.config.get('processed_dir', 'data/processed')
        bm25_path = os.path.join(store_dir, "bm25_index.pkl")
        corpus_path = os.path.join(store_dir, "bm25_corpus.pkl")
        metadata_path = os.path.join(store_dir, "bm25_metadata.pkl")
        
        if not all(os.path.exists(p) for p in [bm25_path, corpus_path, metadata_path]):
            return False
        
        try:
            with open(bm25_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
            
            with open(corpus_path, 'rb') as f:
                self.bm25_corpus = pickle.load(f)
            
            with open(metadata_path, 'rb') as f:
                self.bm25_metadata = pickle.load(f)
            
            print("BM25 index loaded")
            return True
        
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            return False
    
    def semantic_search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[SearchResult]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        # Load vector store if needed
        if self.vector_store.get_stats()['total_vectors'] == 0:
            if not self.vector_store.load("default"):
                print("No semantic index found. Please index some data first.")
                return []
        
        # Get query embedding
        query_embedding = self.embedder.encode_query(query)
        
        # Search
        results = self.vector_store.search(query_embedding, k, threshold)
        
        # Convert to SearchResult objects
        search_results = []
        for text, score, metadata in results:
            search_results.append(SearchResult(
                text=text,
                score=score,
                metadata=metadata,
                search_type="semantic"
            ))

        # Apply additional ranking for individual messages
        search_results = self._rerank_individual_messages(query, search_results)

        return search_results

    def _rerank_individual_messages(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Apply additional ranking for individual message results."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for result in results:
            # Boost score for exact word matches
            text_lower = result.text.lower()
            text_words = set(text_lower.split())

            # Calculate word overlap
            word_overlap = len(query_words.intersection(text_words))
            if word_overlap > 0:
                overlap_boost = word_overlap / len(query_words) * 0.1
                result.score += overlap_boost

            # Boost for exact phrase matches
            if query_lower in text_lower:
                result.score += 0.15

            # Boost for messages with single sender (more focused)
            if 'message_count' in result.metadata and result.metadata['message_count'] == 1:
                result.score += 0.05

            # Penalize very long messages (likely concatenated)
            if len(result.text) > 200:
                result.score -= 0.05

        # Re-sort by updated scores
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def bm25_search(
        self,
        query: str,
        k: int = 5
    ) -> List[SearchResult]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results
        """
        # Try new BM25Store first
        if self.bm25_store.get_stats()['total_documents'] > 0:
            results = self.bm25_store.search(
                query, 
                k=k, 
                min_score=self.config.get('bm25_min_score', 0.0)
            )
            
            search_results = []
            for text, score, metadata in results:
                search_results.append(SearchResult(
                    text=text,
                    score=score,
                    metadata=metadata,
                    search_type="bm25"
                ))
            
            return search_results
        
        # Fallback to legacy BM25 index
        if self.bm25_index is None:
            if not self._load_bm25_index():
                print("No BM25 index found. Please index some data first.")
                return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        search_results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                search_results.append(SearchResult(
                    text=self.bm25_corpus[idx],
                    score=float(scores[idx]),
                    metadata=self.bm25_metadata[idx],
                    search_type="bm25"
                ))
        
        return search_results
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        semantic_threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and BM25 results.
        
        Args:
            query: Search query
            k: Number of results to return
            semantic_weight: Weight for semantic scores
            bm25_weight: Weight for BM25 scores
            semantic_threshold: Minimum semantic similarity threshold
            
        Returns:
            List of search results ranked by combined score
        """
        # Get results from both methods
        semantic_results = self.semantic_search(query, k * 2, semantic_threshold)
        bm25_results = self.bm25_search(query, k * 2)
        
        # Normalize scores to [0, 1] range
        if semantic_results:
            max_semantic = max(r.score for r in semantic_results)
            for result in semantic_results:
                result.score = result.score / max_semantic if max_semantic > 0 else 0
        
        if bm25_results:
            max_bm25 = max(r.score for r in bm25_results)
            for result in bm25_results:
                result.score = result.score / max_bm25 if max_bm25 > 0 else 0
        
        # Combine results
        combined_scores = {}
        
        # Add semantic results
        for result in semantic_results:
            chunk_id = result.metadata.get('chunk_id', result.text[:50])
            combined_scores[chunk_id] = {
                'result': result,
                'semantic_score': result.score,
                'bm25_score': 0.0
            }
        
        # Add BM25 results
        for result in bm25_results:
            chunk_id = result.metadata.get('chunk_id', result.text[:50])
            if chunk_id in combined_scores:
                combined_scores[chunk_id]['bm25_score'] = result.score
            else:
                combined_scores[chunk_id] = {
                    'result': result,
                    'semantic_score': 0.0,
                    'bm25_score': result.score
                }
        
        # Calculate combined scores
        final_results = []
        for chunk_id, data in combined_scores.items():
            combined_score = (
                semantic_weight * data['semantic_score'] +
                bm25_weight * data['bm25_score']
            )
            
            result = data['result']
            result.score = combined_score
            result.search_type = "hybrid"
            final_results.append(result)
        
        # Sort by combined score and return top-k
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:k]
    
    def search(
        self,
        query: str,
        mode: str = "semantic",
        k: int = 5,
        threshold: float = 0.7,
        **kwargs
    ) -> List[SearchResult]:
        """
        Unified search interface.
        
        Args:
            query: Search query
            mode: Search mode ('semantic', 'bm25', 'hybrid')
            k: Number of results to return
            threshold: Minimum similarity threshold (for semantic search)
            **kwargs: Additional arguments for specific search modes
            
        Returns:
            List of search results
        """
        try:
            if mode == "semantic":
                return self.semantic_search(query, k, threshold=threshold, **kwargs)
            elif mode == "bm25":
                # BM25 doesn't use threshold, only pass k
                return self.bm25_search(query, k)
            elif mode == "hybrid":
                # Pass threshold for semantic component of hybrid search
                return self.hybrid_search(query, k, semantic_threshold=threshold, **kwargs)
            else:
                raise ValueError(f"Unknown search mode: {mode}")
        except Exception as e:
            print(f"Search error in {mode} mode: {e}")
            # Return empty list instead of raising to prevent 500 errors
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed data."""
        # Try to load vector store if it's empty
        if self.vector_store.get_stats()['total_vectors'] == 0:
            self.vector_store.load("default")

        vector_stats = self.vector_store.get_stats()

        # Get BM25 count - prioritize new BM25Store
        bm25_count = 0
        bm25_stats = self.bm25_store.get_stats()
        if bm25_stats['total_documents'] > 0:
            bm25_count = bm25_stats['total_documents']
        elif self.bm25_index is not None:
            bm25_count = len(self.bm25_corpus)
        elif self._load_bm25_index():
            bm25_count = len(self.bm25_corpus)

        return {
            'semantic_vectors': vector_stats['total_vectors'],
            'bm25_documents': bm25_count,
            'embedding_dimension': vector_stats['dimension'],
            'index_type': vector_stats['index_type']
        }
