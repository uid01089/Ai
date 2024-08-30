from __future__ import annotations
from typing import List
from Ai.AiChatSystemIf import AiChatSystemIf
from Ai.AiContextIf import AiContextIf
from Ai.AiModelIf import ChatResponse, Prompt


class AiChatSystem(AiChatSystemIf):
    """
    AiChatSystem is a class that coordinates AI chat interactions.
    It maintains chat history and interfaces with AI models to process prompts and generate responses.

    Attributes:
        context (AiContextIf): The contextual interface for AI models.
        history (List[Prompt]): The history of chat prompts.
    """

    def __init__(self, context: AiContextIf) -> None:
        """
        Initializes the AiChatSystem with a given context.

        Args:
            context (AiContextIf): The context interface containing AI model specifics.
        """
        self.context = context
        self.history: List[Prompt] = []

    def clear(self) -> AiChatSystem:
        """
        Clears the chat history.

        Returns:
            AiChatSystem: The instance of the AiChatSystem itself.
        """
        self.history.clear()
        return self

    def append(self, prompt: Prompt) -> AiChatSystem:
        """
        Appends a new prompt to the chat history.

        Args:
            prompt (Prompt): The prompt to be added to the history.

        Returns:
            AiChatSystem: The instance of the AiChatSystem itself.
        """
        self.history.append(prompt)
        return self

    def chat(self, prompt: Prompt) -> ChatResponse:
        """
        Sends a prompt to the AI model and returns the chat response.

        Args:
            prompt (Prompt): The prompt to be sent to the AI model.

        Returns:
            ChatResponse: The response received from the AI model.
        """
        self.history.append(prompt)
        response = self.context.getAiModel().chat(self.history)
        self.history.append(response['prompt'])
        return response

    def getHistory(self) -> List[Prompt]:
        """
        Retrieves the chat history.

        Returns:
            List[Prompt]: The list of prompts representing the chat history.
        """
        return self.history
