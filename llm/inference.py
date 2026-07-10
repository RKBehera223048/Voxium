from __future__ import annotations

import logging
import asyncio
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ChatLLM:
    """
    Core chat completion interface wrapping a llama-cpp-python Llama instance.
    Provides async methods for standard chat completion and tool-calling completion.
    """
    
    def __init__(self, model_instance: Any):
        """
        Initializes the ChatLLM with an active Llama model instance.
        
        Args:
            model_instance: The initialized llama_cpp.Llama object.
        """
        self.model = model_instance
        logger.debug("Initialized ChatLLM with provided model instance.")

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Asynchronously generates a chat completion response.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries containing 'role' and 'content'.
            temperature (float): Sampling temperature (default: 0.7).
            max_tokens (int): Maximum number of tokens to generate (default: 1024).
            
        Returns:
            str: The generated response content.
        """
        logger.info("Starting async chat completion.")
        
        def _run_completion() -> str:
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            try:
                return response["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected completion response structure: {response}")
                raise ValueError("Failed to parse completion response.") from e

        # Run the blocking generation in a thread
        result = await asyncio.to_thread(_run_completion)
        logger.info("Finished async chat completion.")
        return result

    async def complete_with_tools(
        self,
        messages: List[Dict[str, str]],
        tool_definitions: List[Dict[str, Any]],
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Asynchronously generates a completion using tool calling.
        Expects the underlying model to support tool calling formatting.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries.
            tool_definitions (List[Dict[str, Any]]): The tool schema definitions.
            temperature (float): Sampling temperature (default: 0.1 for high determinism).
            
        Returns:
            Dict[str, Any]: A dictionary representing the tool call(s) or standard response.
                            The format follows the OpenAI standard for tool calls.
        """
        logger.info(f"Starting async tool-based completion with {len(tool_definitions)} tools.")
        
        def _run_tool_completion() -> Any:
            response = self.model.create_chat_completion(
                messages=messages,
                tools=tool_definitions,
                tool_choice="auto",
                temperature=temperature,
                stream=False
            )
            return response
            
        response = await asyncio.to_thread(_run_tool_completion)
        
        try:
            message = response["choices"][0]["message"]
            logger.info("Finished async tool-based completion.")
            return message
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected completion response structure: {response}")
            raise ValueError("Failed to parse tool completion response.") from e
