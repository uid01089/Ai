            
from typing import List
from litellm import completion, token_counter, get_max_tokens


from Ai.AiContextIf import AiContextIf
from Ai.AiModelIf import AiModelIf, ChatResponse, Prompt

# https://docs.litellm.ai/docs/#basic-usage
# https://docs.litellm.ai/docs/completion/token_usage
# https://docs.litellm.ai/docs/providers/


class AiModel(AiModelIf):
    """
    AiModel class implements the AiModelIf interface to perform AI-based operations
    using the given model and context interfaces. This includes generating chat
    responses, counting tokens in prompts, and retrieving the maximum possible
    number of tokens for the model.
    """

    def __init__(self, model: str, context: AiContextIf) -> None:
        """
        Initialize the AiModel with the specified model and context.

        Parameters:
        model (str): The name or identifier of the AI model.
        context (AiContextIf): The context interface to be used with the model.
        """
        self.model = model
        self.context = context

    def chat(self, prompts: List[Prompt]) -> ChatResponse:
        """
        Generate a chat response based on the provided prompts.

        Parameters:
        prompts (List[Prompt]): A list of prompts to send to the AI model.

        Returns:
        ChatResponse: The chat response including the assistant's message content
                      and token usage information.
        """
        response = completion(
            model=self.model,
            messages=prompts
        )

        return {'prompt': {'role': 'assistant',
                           'content': response.choices[0].message.content},
                'completionTokens': response.model_extra['usage'].completion_tokens,
                'promptTokens': response.model_extra['usage'].prompt_tokens,
                'totalTokens': response.model_extra['usage'].total_tokens
                }

    def getNumberOfTokens(self, prompts: List[Prompt]) -> int:
        """
        Calculate the total number of tokens in the provided prompts for the model.

        Parameters:
        prompts (List[Prompt]): A list of prompts whose tokens need to be counted.

        Returns:
        int: The total number of tokens in the prompts.
        """
        string = ""
        for prompt in prompts:
            string = string + prompt['content']

        return token_counter(self.model, string)

    def getPossibleNumberTokens(self) -> int:
        """
        Get the maximum possible number of tokens for the model.

        Returns:
        int: The maximum number of tokens the model can handle.
        """
        return get_max_tokens(self.model)