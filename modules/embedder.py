"""
Embedding module for ChatWhiz using Instructor-Large model.
Provides semantic embeddings for chat messages with instruction-tuned approach.

This module includes fixes for:
- SentenceTransformer._target_device deprecation warning
- Proper device handling for CPU/GPU usage
- Comprehensive warning suppression for cleaner output
"""

import os
import pickle
import hashlib
import warnings
import logging
from typing import List, Union, Optional, Any
import numpy as np

# Comprehensive warning suppression for the _target_device deprecation
warnings.filterwarnings("ignore", message=".*_target_device.*deprecated.*")
warnings.filterwarnings("ignore", message=".*_target_device.*")
warnings.filterwarnings("ignore", message=".*property 'device'.*has no setter.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="sentence_transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")

# Also suppress logging warnings from sentence_transformers
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Import with warning suppression
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from InstructorEmbedding import INSTRUCTOR


def suppress_warnings(func):
    """Decorator to suppress warnings during function execution."""
    def wrapper(*args, **kwargs):
        import sys
        import io

        # Capture both warnings and stderr output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Temporarily redirect stderr to suppress print warnings
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()

            try:
                result = func(*args, **kwargs)
            finally:
                # Restore stderr
                captured_stderr = sys.stderr.getvalue()
                sys.stderr = old_stderr

                # Only print stderr if it doesn't contain known warnings
                if captured_stderr and not any(warning in captured_stderr for warning in [
                    "_target_device", "property 'device'", "has no setter"
                ]):
                    print(captured_stderr, file=sys.stderr, end='')

            return result
    return wrapper


class InstructorEmbedder:
    """
    Wrapper for Instructor-Large embedding model with caching support.
    """

    def __init__(
        self,
        model_name: str = "hkunlp/instructor-large",
        instruction: str = "Represent the chat message for semantic search:",
        cache_dir: str = "data/cache",
        device: Optional[str] = None
    ):
        """
        Initialize the Instructor embedder.

        Args:
            model_name: HuggingFace model name for Instructor
            instruction: Instruction text for embedding
            cache_dir: Directory to cache embeddings
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_name = model_name
        self.instruction = instruction
        self.cache_dir = cache_dir
        self.device = device
        self.model = None

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

    def _get_device(self):
        """Get the appropriate device for the model."""
        if self.device is not None and self.device.lower() not in ['auto', 'none']:
            return self.device

        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'
        
    @suppress_warnings
    def _load_model(self):
        """Lazy load the model to save memory."""
        if self.model is None:
            print(f"Loading Instructor model: {self.model_name}")
            device = self._get_device()
            print(f"Using device: {device}")

            # Temporarily suppress the specific deprecation warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*_target_device.*deprecated.*")

                try:
                    # Load the INSTRUCTOR model with proper device handling
                    self.model = INSTRUCTOR(self.model_name, device=device)

                    # Post-loading device fix for the underlying SentenceTransformer
                    self._fix_device_deprecation(device)

                    print("Model loaded successfully!")

                except Exception as e:
                    print(f"Error loading model with device {device}: {e}")
                    # Fallback to default loading without device specification
                    try:
                        print("Attempting fallback loading without device specification...")
                        self.model = INSTRUCTOR(self.model_name)
                        self._fix_device_deprecation('cpu')  # Default to CPU for fallback
                        print("Model loaded with fallback method!")
                    except Exception as fallback_error:
                        print(f"Fallback loading also failed: {fallback_error}")
                        raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

    def _fix_device_deprecation(self, device: str):
        """Fix the deprecated _target_device warning by properly setting the device."""
        try:
            import torch
            device_obj = torch.device(device)

            # The INSTRUCTOR model is itself a SentenceTransformer
            # Try to access the underlying SentenceTransformer attributes
            if self.model is not None and hasattr(self.model, '_target_device'):
                try:
                    # Set _target_device to match the actual device to prevent warnings
                    self.model._target_device = device_obj
                except (AttributeError, TypeError):
                    # If we can't set it, that's fine - the warning is just cosmetic
                    pass

            # Ensure all modules are on the correct device
            if self.model is not None and hasattr(self.model, 'to'):
                self.model.to(device_obj)

            # Try to set device attribute if it exists and is settable
            if self.model is not None and hasattr(self.model, 'device'):
                try:
                    # Some versions have a settable device attribute
                    if hasattr(type(self.model).device, 'fset') or not hasattr(type(self.model), 'device'):
                        self.model.device = device_obj
                except (AttributeError, TypeError):
                    # If device is a property without setter, that's fine
                    pass

        except Exception as e:
            # Suppress the warning message since it's not critical
            pass
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for a list of texts."""
        combined_text = f"{self.instruction}|{self.model_name}|{'|'.join(texts)}"
        return hashlib.md5(combined_text.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embeddings from cache if available."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cache {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embeddings: np.ndarray):
        """Save embeddings to cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_key}: {e}")
    
    @suppress_warnings
    def encode(
        self,
        texts: Union[str, List[str]],
        use_cache: bool = True,
        batch_size: int = 32,
        progress_callback: Optional[Any] = None
    ) -> np.ndarray:
        """
        Encode texts into embeddings using Instructor model.
        
        Args:
            texts: Single text or list of texts to encode
            use_cache: Whether to use caching
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(texts)
            cached_embeddings = self._load_from_cache(cache_key)
            if cached_embeddings is not None:
                print(f"Loaded {len(texts)} embeddings from cache")
                return cached_embeddings
        
        # Load model if needed
        self._load_model()

        # Prepare instruction-text pairs
        instruction_text_pairs = [[self.instruction, text] for text in texts]

        # Encode in batches to manage memory with warning suppression
        all_embeddings = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*_target_device.*deprecated.*")

            for i in range(0, len(instruction_text_pairs), batch_size):
                batch = instruction_text_pairs[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(instruction_text_pairs) + batch_size - 1)//batch_size

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(batch_num, total_batches, len(batch))
                else:
                    print(f"Encoding batch {batch_num}/{total_batches}")

                batch_embeddings = self.model.encode(batch)
                all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        
        # Save to cache
        if use_cache:
            self._save_to_cache(cache_key, embeddings)
            print(f"Saved {len(texts)} embeddings to cache")
        
        return embeddings
    
    @suppress_warnings
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query for search.

        Args:
            query: Search query text

        Returns:
            Query embedding as numpy array
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*_target_device.*deprecated.*")
            return self.encode([query], use_cache=False)[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model."""
        # Check if we have a cached dimension first
        dimension_cache_path = os.path.join(self.cache_dir, f"{self.model_name.replace('/', '_')}_dimension.txt")

        if os.path.exists(dimension_cache_path):
            try:
                with open(dimension_cache_path, 'r') as f:
                    cached_dim = int(f.read().strip())
                    return cached_dim
            except (ValueError, IOError):
                pass  # Fall back to computing dimension

        self._load_model()
        # Test with a dummy text to get dimension
        test_embedding = self.encode(["test"], use_cache=False)
        dimension = test_embedding.shape[1]

        # Cache the dimension for future use
        try:
            with open(dimension_cache_path, 'w') as f:
                f.write(str(dimension))
        except IOError:
            pass  # Not critical if we can't cache

        return dimension

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        self._load_model()
        info = {
            'model_name': self.model_name,
            'device': self._get_device(),
            'instruction': self.instruction,
            'embedding_dimension': self.get_embedding_dimension()
        }

        # Get additional info from the INSTRUCTOR model (which is a SentenceTransformer)
        if self.model is not None and hasattr(self.model, 'get_sentence_embedding_dimension'):
            info['sentence_transformer_dimension'] = self.model.get_sentence_embedding_dimension()

        if self.model is not None and hasattr(self.model, '_modules'):
            info['model_modules'] = list(self.model._modules.keys())

            # Get architecture info from the transformer module
            if '0' in self.model._modules:
                transformer_module = self.model._modules['0']
                if hasattr(transformer_module, 'auto_model'):
                    info['transformer_architecture'] = type(transformer_module.auto_model).__name__

        if self.model is not None and hasattr(self.model, 'tokenizer'):
            info['tokenizer_type'] = type(self.model.tokenizer).__name__

        return info
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        import glob
        cache_files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
            except Exception as e:
                print(f"Warning: Failed to remove cache file {cache_file}: {e}")
        print(f"Cleared {len(cache_files)} cache files")


def create_embedder_from_config(config: dict) -> InstructorEmbedder:
    """
    Create an InstructorEmbedder from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured InstructorEmbedder instance
    """
    return InstructorEmbedder(
        model_name=config.get('embedding_model', 'hkunlp/instructor-large'),
        instruction=config.get('instruction', 'Represent the chat message for semantic search:'),
        cache_dir=config.get('cache_dir', 'data/cache'),
        device=config.get('device')
    )
