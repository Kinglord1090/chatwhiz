"""
LLM integration module for ChatWhiz supporting OpenAI and Ollama for RAG functionality.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# SearchResult will be passed as parameter, no need to import


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for RAG responses."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if self.api_key and OPENAI_AVAILABLE:
            openai.api_key = self.api_key
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return OPENAI_AVAILABLE and self.api_key is not None
    
    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """Generate response using OpenAI API."""
        if not self.is_available():
            raise RuntimeError("OpenAI provider is not available")
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model: Ollama model name
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
    
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """Generate response using Ollama API."""
        if not self.is_available():
            raise RuntimeError("Ollama server is not available")
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            return result.get("response", "").strip()
        
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")


class RAGSystem:
    """
    Retrieval-Augmented Generation system for ChatWhiz.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        max_context_length: int = 2000
    ):
        """
        Initialize RAG system.
        
        Args:
            llm_provider: LLM provider instance
            max_context_length: Maximum context length for prompts
        """
        self.llm_provider = llm_provider
        self.max_context_length = max_context_length
    
    def format_context(self, search_results: List) -> str:
        """
        Format search results into context for the LLM.
        
        Args:
            search_results: List of search results
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant chat messages found."
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(search_results, 1):
            # Format each result
            result_text = f"Message {i} (Score: {result.score:.3f}):\n{result.text}\n"
            
            # Check if adding this result would exceed max length
            if current_length + len(result_text) > self.max_context_length:
                break
            
            context_parts.append(result_text)
            current_length += len(result_text)
        
        return "\n".join(context_parts)
    
    def create_prompt(
        self,
        query: str,
        context: str,
        prompt_template: Optional[str] = None
    ) -> str:
        """
        Create a prompt for the LLM using query and context.
        
        Args:
            query: User query
            context: Retrieved context
            prompt_template: Custom prompt template
            
        Returns:
            Formatted prompt
        """
        if prompt_template is None:
            prompt_template = """Based on the following chat messages, answer the question below. If the information is not available in the chat messages, say so clearly.

Chat Messages:
{context}

Question: {query}

Answer:"""
        
        return prompt_template.format(context=context, query=query)
    
    def generate_answer(
        self,
        query: str,
        search_results: List,
        prompt_template: Optional[str] = None,
        **llm_kwargs
    ) -> Dict[str, Any]:
        """
        Generate an answer using RAG.
        
        Args:
            query: User query
            search_results: Retrieved search results
            prompt_template: Custom prompt template
            **llm_kwargs: Additional arguments for LLM
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.llm_provider.is_available():
            return {
                "answer": "LLM provider is not available. Showing search results only.",
                "context_used": len(search_results),
                "error": "LLM unavailable"
            }
        
        # Format context
        context = self.format_context(search_results)
        
        # Create prompt
        prompt = self.create_prompt(query, context, prompt_template)
        
        try:
            # Generate response
            answer = self.llm_provider.generate_response(prompt, **llm_kwargs)
            
            return {
                "answer": answer,
                "context_used": len(search_results),
                "prompt_length": len(prompt),
                "search_results": search_results
            }
        
        except Exception as e:
            return {
                "answer": f"Error generating response: {e}",
                "context_used": len(search_results),
                "error": str(e),
                "search_results": search_results
            }
    
    def summarize_conversation(
        self,
        search_results: List,
        focus: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Summarize a conversation from search results.
        
        Args:
            search_results: Retrieved search results
            focus: Optional focus for the summary
            
        Returns:
            Dictionary with summary and metadata
        """
        if not search_results:
            return {"summary": "No messages found to summarize."}
        
        context = self.format_context(search_results)
        
        if focus:
            prompt = f"""Summarize the following chat conversation with a focus on: {focus}

Chat Messages:
{context}

Summary:"""
        else:
            prompt = f"""Summarize the following chat conversation, highlighting the main topics and key points discussed.

Chat Messages:
{context}

Summary:"""
        
        try:
            summary = self.llm_provider.generate_response(prompt, max_tokens=300)
            
            return {
                "summary": summary,
                "messages_summarized": len(search_results),
                "focus": focus
            }
        
        except Exception as e:
            return {
                "summary": f"Error generating summary: {e}",
                "error": str(e)
            }


def create_llm_provider(config: Dict[str, Any]) -> Optional[LLMProvider]:
    """
    Create an LLM provider based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LLM provider instance or None if disabled
    """
    provider_type = config.get('llm_provider', 'none').lower()
    
    if provider_type == 'none':
        return None
    
    elif provider_type == 'openai':
        api_key = config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        model = config.get('openai_model', 'gpt-3.5-turbo')
        return OpenAIProvider(api_key=api_key, model=model)
    
    elif provider_type == 'ollama':
        model = config.get('ollama_model', 'llama2')
        url = config.get('ollama_url', 'http://localhost:11434')
        return OllamaProvider(model=model, base_url=url)
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider_type}")


def create_rag_system(config: Dict[str, Any]) -> Optional[RAGSystem]:
    """
    Create a RAG system based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RAG system instance or None if LLM is disabled
    """
    llm_provider = create_llm_provider(config)
    
    if llm_provider is None:
        return None
    
    max_context = config.get('max_context_length', 2000)
    return RAGSystem(llm_provider, max_context)
