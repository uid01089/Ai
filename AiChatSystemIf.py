from abc import ABC, abstractmethod

from Ai.AiModelIf import ChatResponse, Prompt


class AiChatSystemIf(ABC):
    @abstractmethod
    def chat(self, prompt: Prompt) -> ChatResponse:
        pass
