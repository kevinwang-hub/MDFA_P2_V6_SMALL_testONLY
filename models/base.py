"""Abstract base classes for model clients."""

from abc import ABC, abstractmethod


class VLMClient(ABC):
    """Vision-Language Model client."""

    @abstractmethod
    def generate(
        self,
        image: str,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """
        Args:
            image: base64-encoded image string or file path
            system: system prompt
            user: user prompt
            temperature: sampling temperature
            max_tokens: max output tokens
        Returns:
            Raw text response from model
        """
        pass


class LLMClient(ABC):
    """Text-only LLM client."""

    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> str:
        pass
