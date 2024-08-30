from abc import ABC, abstractmethod
from typing import List, Literal, TypedDict


class Prompt(TypedDict):
    """
    Represents a chat message with a specific role and content.
    
    Attributes:
    -----------
    role : Literal['user', 'assistant', 'system']
        The role of the message sender. Response messages always have the role 'assistant'.
    content : str
        The actual content of the message. In case of responses, this may contain message fragments when streaming.
    """
    role: Literal['user', 'assistant', 'system']
    "Assumed role of the message. Response messages always have role 'assistant'."

    content: str
    'Content of the message. Response messages contain message fragments when streaming.'


class ChatResponse(TypedDict):
    """
    Represents the response from a chat model.
    
    Attributes:
    -----------
    prompt : Prompt
        The original prompt that triggered the response.
    completionTokens : int
        The number of tokens used to generate the response.
    promptTokens : int
        The number of tokens in the given prompt.
    totalTokens : int
        The total number of tokens consumed (prompt + completion).
    """
    prompt: Prompt
    completionTokens: int
    promptTokens: int
    totalTokens: int


class AiModelIf(ABC):
    """
    Abstract Base Class for AI chat models. Defines the interface for interacting with the model.
    """
    @abstractmethod
    def chat(self, prompts: List[Prompt]) -> ChatResponse:
        """
        Generates a chat response based on the provided prompts.
        
        Parameters:
        -----------
        prompts : List[Prompt]
            A list of prompts where each prompt contains a role and content.
        
        Returns:
        --------
        ChatResponse
            The response from the AI model including the original prompt and token details.
        """
        pass

    @abstractmethod
    def getNumberOfTokens(self, prompts: List[Prompt]) -> int:
        """
        Calculates the number of tokens in the given list of prompts.
        
        Parameters:
        -----------
        prompts : List[Prompt]
            A list of prompts to analyze.
        
        Returns:
        --------
        int
            The total number of tokens in the provided prompts.
        """
        pass

    @abstractmethod
    def getPossibleNumberTokens(self) -> int:
        """
        Returns the maximum number of tokens that the model can handle.
        
        Returns:
        --------
        int
            The maximum number of tokens supported by the AI model.
        """
        pass