from __future__ import annotations

import logging
import asyncio
from typing import List, Dict, Any, Callable, Awaitable, AsyncGenerator, Optional

logger = logging.getLogger(__name__)

async def stream_completion(
    model_instance: Any,
    messages: List[Dict[str, str]],
    on_token_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> AsyncGenerator[str, None]:
    """
    Streams the token generation from a Llama model instance asynchronously.
    
    Args:
        model_instance: The initialized llama_cpp.Llama object.
        messages (List[Dict[str, str]]): The conversation history.
        on_token_callback (Optional[Callable[[str], Awaitable[None]]]): Optional async callback invoked per token.
        temperature (float): Sampling temperature (default: 0.7).
        max_tokens (int): Maximum tokens to generate (default: 1024).
        
    Yields:
        str: Each generated token chunk.
    """
    logger.info("Starting streaming completion.")
    
    # We use an asyncio.Queue to safely pass tokens from the generator thread to the async loop
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    def _generator_thread():
        try:
            generator = model_instance.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in generator:
                try:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        content = delta["content"]
                        if content:
                            loop.call_soon_threadsafe(queue.put_nowait, content)
                except (KeyError, IndexError):
                    continue
        except Exception as e:
            logger.error(f"Streaming error in thread: {e}")
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # Sentinel to signal completion

    # Start the blocking generation in a background thread
    asyncio.create_task(asyncio.to_thread(_generator_thread))
    
    try:
        while True:
            item = await queue.get()
            if item is None:  # Sentinel value
                break
            if isinstance(item, Exception):
                raise item
            
            if on_token_callback:
                await on_token_callback(item)
                
            yield item
    finally:
        logger.info("Finished streaming completion.")
