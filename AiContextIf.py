

from abc import ABC, abstractmethod

from Ai.AiChatSystemIf import AiChatSystemIf
from Ai.AiModelIf import AiModelIf


class AiContextIf(ABC):

    @abstractmethod
    def getAiModel(self) -> AiModelIf:
        pass

    @abstractmethod
    def createChat(self) -> AiChatSystemIf:
        pass
