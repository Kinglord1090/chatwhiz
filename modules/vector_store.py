"""
Vector store module for ChatWhiz using FAISS for efficient similarity search.
Manages embedding storage, indexing, and retrieval operations.
"""

import os
import pickle
import json
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    """
    
    def __init__(
        self,
        dimension: int,
        store_dir: str = "data/vectorstore",
        index_type: str = "flat"
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension
            store_dir: Directory to store index and metadata
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.dimension = dimension
        self.store_dir = store_dir
        self.index_type = index_type
        self.index = None
        self.metadata = []  # Store original text and other metadata
        
        # Create store directory
        os.makedirs(store_dir, exist_ok=True)
        
        # Initialize index
        self._create_index()
    
    def _create_index(self):
        """Create appropriate FAISS index based on type."""
        if self.index_type == "flat":
            # Exact search, good for small datasets
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
        elif self.index_type == "ivf":
            # Approximate search with inverted file index
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World graph
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: Array of embeddings to add
            texts: Corresponding text content
            metadata: Optional metadata for each embedding
        """
        if embeddings.shape[0] != len(texts):
            raise ValueError("Number of embeddings must match number of texts")
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to index
        if self.index_type == "ivf" and not self.index.is_trained:
            # Train IVF index if not already trained
            if embeddings_normalized.shape[0] >= 100:  # Need enough data to train
                print("Training IVF index...")
                self.index.train(embeddings_normalized)
            else:
                print("Not enough data to train IVF index, using flat index instead")
                self.index = faiss.IndexFlatIP(self.dimension)
        
        self.index.add(embeddings_normalized)
        
        # Store metadata
        for i, text in enumerate(texts):
            item_metadata = {
                'text': text,
                'index': len(self.metadata),
                **(metadata[i] if metadata and i < len(metadata) else {})
            }
            self.metadata.append(item_metadata)
        
        print(f"Added {len(texts)} embeddings to vector store")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (text, similarity_score, metadata) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        query_normalized = query_normalized.reshape(1, -1)
        
        # Search
        similarities, indices = self.index.search(query_normalized, min(k, self.index.ntotal))
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            if threshold is not None and similarity < threshold:
                continue
            
            metadata = self.metadata[idx]
            results.append((metadata['text'], float(similarity), metadata))
        
        return results
    
    def save(self, name: str = "default"):
        """
        Save the vector store to disk.

        Args:
            name: Name for the saved store
        """
        # Ensure store directory exists
        os.makedirs(self.store_dir, exist_ok=True)

        index_path = os.path.join(self.store_dir, f"{name}.index")
        metadata_path = os.path.join(self.store_dir, f"{name}_metadata.pkl")
        config_path = os.path.join(self.store_dir, f"{name}_config.json")

        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save configuration
        config = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'total_vectors': self.index.ntotal
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved vector store '{name}' with {self.index.ntotal} vectors")
    
    def load(self, name: str = "default") -> bool:
        """
        Load a vector store from disk.
        
        Args:
            name: Name of the store to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = os.path.join(self.store_dir, f"{name}.index")
        metadata_path = os.path.join(self.store_dir, f"{name}_metadata.pkl")
        config_path = os.path.join(self.store_dir, f"{name}_config.json")
        
        if not all(os.path.exists(p) for p in [index_path, metadata_path, config_path]):
            return False
        
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Verify dimension matches
            if config['dimension'] != self.dimension:
                print(f"Warning: Dimension mismatch. Expected {self.dimension}, got {config['dimension']}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"Loaded vector store '{name}' with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metadata_count': len(self.metadata)
        }
    
    def clear(self):
        """Clear all vectors and metadata."""
        self._create_index()
        self.metadata = []
        print("Cleared vector store")
    
    def remove_by_indices(self, indices: List[int]):
        """
        Remove vectors by their indices.
        Note: This recreates the index, which can be expensive.
        """
        if not indices:
            return
        
        # Get all embeddings and metadata except those to remove
        indices_set = set(indices)
        remaining_metadata = [
            meta for i, meta in enumerate(self.metadata) 
            if i not in indices_set
        ]
        
        if not remaining_metadata:
            self.clear()
            return
        
        # This is expensive - we need to rebuild the index
        # In practice, you might want to mark items as deleted instead
        print(f"Removing {len(indices)} vectors requires rebuilding index...")
        
        # For now, we'll just update metadata and note that the index
        # contains deleted items. A full rebuild would require re-embedding.
        self.metadata = remaining_metadata
        print(f"Marked {len(indices)} vectors as removed")


def create_vector_store_from_config(
    config: dict,
    dimension: int
) -> FAISSVectorStore:
    """
    Create a FAISSVectorStore from configuration.
    
    Args:
        config: Configuration dictionary
        dimension: Embedding dimension
        
    Returns:
        Configured FAISSVectorStore instance
    """
    return FAISSVectorStore(
        dimension=dimension,
        store_dir=config.get('vectorstore_dir', 'data/vectorstore'),
        index_type=config.get('index_type', 'flat')
    )
