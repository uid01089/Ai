from Ai.AiChatSystem import AiChatSystem
from Ai.AiChatSystemIf import AiChatSystemIf
from Ai.AiContextIf import AiContextIf
from Ai.AiModel import AiModel
from Ai.AiModelIf import AiModelIf


class AiContext(AiContextIf):
    def __init__(self, model: str) -> None:
        self.aiModel = AiModel(model, self)

    def getAiModel(self) -> AiModelIf:
        return self.aiModel

    def createChat(self) -> AiChatSystemIf:
        return AiChatSystem(self)
