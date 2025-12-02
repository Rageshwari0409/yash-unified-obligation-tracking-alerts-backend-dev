"""
LiteLLM Client for Obligation Tracking System.
Provides unified interface for LLM operations using LiteLLM with Gemini backend.
"""

import os
import yaml
import logging
from typing import Optional, Dict, Any, List
import litellm
from litellm import completion
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.utils.obs import LLMUsageTracker

logger = logging.getLogger(__name__)


class LiteLLMClient:
    """LiteLLM client with Gemini integration for contract obligation extraction."""

    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize LiteLLM client with configuration.

        Args:
            config_path: Path to model configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        model_config = self.config['models']['gemini']

        # Set API key from environment
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        # Configure LiteLLM
        os.environ["GEMINI_API_KEY"] = self.api_key

        # Model settings
        self.model_name = model_config['model_name']
        self.generation_config = {
            'temperature': model_config['temperature'],
            'max_tokens': model_config['max_tokens'],
            'top_p': model_config['top_p'],
        }

        # Embedding model settings (from config)
        embedding_config = self.config['models'].get('embedding', {})
        self.embedding_model_name = f"models/{embedding_config.get('model_name', 'text-embedding-004')}"
        self.embedding_dimension = embedding_config.get('dimension', 768)

        # Initialize Google Generative AI Embeddings
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            google_api_key=self.api_key,
            model=self.embedding_model_name
        )

        # API settings
        api_settings = self.config['api_settings']
        self.timeout = api_settings['timeout']
        self.max_retries = api_settings['max_retries']

        logger.info(f"LiteLLM client initialized with model: {self.model_name}")
        logger.info(f"Embedding model initialized: {self.embedding_model_name}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        token_tracker:LLMUsageTracker = None,
        llm_params: Dict[str, Any] = None
    ) -> str:
        """
        Generate text based on prompt using LiteLLM.

        Args:
            prompt: User prompt for generation
            system_prompt: Optional system prompt for context
            token_tracker: Optional LLMUsageTracker for tracking usage
            llm_params: LLM parameters (model, api_key, temperature, max_tokens)

        Returns:
            Generated text response
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = completion(
                **llm_params,
                messages=messages,
                timeout=self.timeout,
                num_retries=self.max_retries,
            )


            token_tracker.track_response(response)

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_with_markdown(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate Markdown response from prompt.

        Args:
            prompt: User prompt expecting Markdown response
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Markdown string response
        """
        # Add Markdown instruction to prompt
        markdown_prompt = f"{prompt}\n\nRespond with valid Markdown format only."

        return self.generate(
            prompt=markdown_prompt,
            system_prompt=system_prompt,
            temperature=kwargs.get('temperature', 0.3),
            **{k: v for k, v in kwargs.items() if k != 'temperature'}
        )
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using Google Generative AI.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            return self.embedding_model.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def get_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            return self.embedding_model.embed_query(text)
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            raise

