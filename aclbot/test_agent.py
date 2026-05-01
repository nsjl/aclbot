# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

import os
import asyncio
import logging
from datetime import datetime
import atexit
import traceback
import json
import re
import dotenv
import torch

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai import PromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)

from paper_plugin import PaperPlugin
from paper_service import PaperSearchService
from util import OpenAICostTracker
from plotter import plot  # Custom plotter module

dotenv.load_dotenv()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Configuration ===
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
SERVICE_ID = "paper_search"
HISTORY_OUT_DIR = 'data/recorded_chats'
MAX_ASSISTANT_CHARS = 5000

def truncate_result(res_str: str, max_chars: int = 5000) -> str:
    if len(res_str) > max_chars:
        return res_str[:max_chars] + "\n\n[Truncated due to length]"
    return res_str

def format_cypher_result(cypher_result):
    if isinstance(cypher_result, list):
        return "\n".join(json.dumps(row, ensure_ascii=False) for row in cypher_result)
    return str(cypher_result)


def extract_cypher(text):
    print("------ ORIGINAL TEXT ------")
    print(text)
    print("------ END TEXT ------")

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n').strip()

    # Handle ```cypher ... ``` or ```json with embedded cypher_query```
    code_block = re.search(r"```(?:cypher|json)?\s*\n([\s\S]+?)```", text, re.IGNORECASE)
    if code_block:
        print("Matched code block")
        block_content = code_block.group(1).strip()

        # Try parsing JSON inside the block
        try:
            json_obj = json.loads(block_content)
            if "cypher_query" in json_obj:
                print("Extracted from JSON key: cypher_query")
                return json_obj["cypher_query"].strip()
        except json.JSONDecodeError:
            pass  # fall through to next step

        # Remove wrapping { "cypher_query": ... } even if not proper JSON
        cypher_inline_match = re.search(r'"cypher_query"\s*:\s*"([^"]+)"', block_content)
        if cypher_inline_match:
            print("Extracted from JSON-style key")
            return cypher_inline_match.group(1).strip()

        # Otherwise treat entire block as Cypher
        return block_content

    # Inline JSON-looking query string
    json_inline_match = re.search(r'^{.*"cypher_query"\s*:\s*"([^"]+?)"}$', text)
    if json_inline_match:
        print("Matched inline JSON-style block")
        return json_inline_match.group(1).strip()

    # Fallback: match an inline Cypher query
    inline = re.search(r"(MATCH\s.*?RETURN\s.*?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if inline:
        print("Matched inline Cypher")
        return inline.group(1).strip().rstrip('"')

    print("No Cypher match found.")
    return None

def slice_chat_history(history: ChatHistory, skip_n=0) -> ChatHistory:
    new_hist = ChatHistory()
    for msg in history.messages[skip_n:]:
        new_hist.add_message(msg)
    return new_hist

# pre 9/3/2026
# def init_kernel() -> Kernel:
#     kernel = Kernel()
#     paper_search_service = PaperSearchService(
#         uri=NEO4J_URI,
#         user=NEO4J_USER,
#         pwd=NEO4J_PASSWORD
#     )
#     kernel.add_plugin(PaperPlugin(paper_search_service), plugin_name='paper_search')
#     kernel.add_service(OpenAIChatCompletion(
#         ai_model_id='gpt-4o',
#         api_key=OPENAI_API_KEY,
#         service_id=SERVICE_ID   
#     ))
#     return kernel


from openai import AsyncOpenAI

def init_kernel() -> Kernel:
    kernel = Kernel()

    paper_search_service = PaperSearchService(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        pwd=NEO4J_PASSWORD
    )

    kernel.add_plugin(PaperPlugin(paper_search_service), plugin_name='paper_search')

    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        timeout=60
    )

    kernel.add_service(OpenAIChatCompletion(
        ai_model_id="gpt-4o-mini",
        async_client=client,
        service_id=SERVICE_ID
    ))

    return kernel

def init_settings(kernel):
    settings: OpenAIChatPromptExecutionSettings = kernel.get_prompt_execution_settings_from_service_id(
        service_id=SERVICE_ID
    )
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto(
        filters={
            "included_plugins": ["paper_search"],
            "included_functions": [
                "get_paper_by_id", "semantic_search", "event_search", "run_cypher_query"
            ]
        }
    )
    # settings.max_tokens = 16300
    settings.max_tokens = 10000
    # settings.temperature = 0
    return settings


def save_chat_history(history, date_and_time):
    try:
        os.makedirs(HISTORY_OUT_DIR, exist_ok=True)
        with open(os.path.join(HISTORY_OUT_DIR, f"{date_and_time}.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps([m.to_dict() for m in history.messages], ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")


async def basic_agent(kernel: Kernel, settings: PromptExecutionSettings, openai_cost_tracker: OpenAICostTracker):
    date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("Starting agent...\n")

    history = ChatHistory()

    chat_completion: OpenAIChatCompletion = kernel.get_service(type=OpenAIChatCompletion)

    # Add intro messages
    # 
    #
    # 
    # 
    # Call get_kg_instructions to enable Cypher queries
    result = await kernel.invoke(
        plugin_name="paper_search",
        function_name="get_kg_instructions",
        arguments=KernelArguments()
    )

    history.add_message({
        "role": "assistant",  # or "function" if you want to distinguish it
        "content": str(result)
    })

    print(str(result))

    initial_message = PaperSearchService.get_initial_assistant_message()
    history.add_message(ChatMessageContent(role='assistant', content=initial_message))

    def save_chat_on_exit():
        save_chat_history(slice_chat_history(history, 2), date_and_time)

    atexit.register(save_chat_on_exit)

    print(f'Assistant > {initial_message}')

    while True:
        try:
            print("------ Ready for user input ------")
            user_input = input("\nUser > ")
            print('')

            if user_input.lower() == "exit":
                break
            if user_input.lower() == "restart":
                save_chat_history(history, date_and_time)
                date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                history.messages = history.messages[:2]  # keep system & intro
                print('\nAll content removed from history. Ready to receive new inputs.\n')
                print(f'\nAssistant > {PaperSearchService.get_initial_assistant_message()}\n')
                continue

            history.add_user_message(user_input)

            for msg in history.messages:
                if hasattr(msg, 'content') and isinstance(msg.content, str) and len(msg.content) > MAX_ASSISTANT_CHARS:
                    msg.content = msg.content[:MAX_ASSISTANT_CHARS] + "\n\n[Truncated early to avoid API error]"

            result = (await chat_completion.get_chat_message_contents(
                chat_history=history,
                settings=settings,
                kernel=kernel,
                arguments=KernelArguments(input="Generate a Cypher query."),
            ))[0]

            print("** Conversation History **")
            for msg in history.messages:
                print(f"{msg.role.upper()}: {msg.content}\n")
            print("** End of History **\n")

            if len(result.content) > MAX_ASSISTANT_CHARS:
                logger.warning(f"Assistant response too long ({len(result.content)} chars). Truncating.")
                result.content = result.content[:MAX_ASSISTANT_CHARS] + "\n\n[Truncated]"

            print("\nAssistant > " + str(result) + "\n")

            # Attempt to extract and run Cypher query
            cypher = extract_cypher(result.content)
            print(cypher)
            if cypher:
                print("[System] Extracted Cypher query:\n", cypher)
                try:
                    args = KernelArguments()
                    args["cypher_query"] = cypher
                    cypher_result = await kernel.invoke(
                        plugin_name="paper_search",
                        function_name="run_cypher_query",
                        arguments=args
                    )

                    print("\nAssistant > Query executed. Generating summary...\n")

                    # history.add_message(ChatMessageContent(
                    #     role="user",
                    #     content=f"Here are the query results:\n{format_cypher_result(cypher_result)}\nPlease summarize them in a human-readable way."
                    # ))

                    summary_prompt = f"""Here are the query results:

                    {format_cypher_result(cypher_result)}

                    Please summarize them in a human-readable way."""
                    history.add_message(ChatMessageContent(role="user", content=summary_prompt))

                    summary = (await chat_completion.get_chat_message_contents(
                        chat_history=history,
                        settings=settings,
                        kernel=kernel,
                        arguments=KernelArguments(),
                    ))[0]

                    print("\nAssistant >", summary.content, "\n")
                    history.add_message(summary)

                    continue  # Skip default response
                except Exception as e:
                    print("[System] Failed to run Cypher query:", e)

            # Try plotting
            plot_item = plot(user_input, result.content)
            if plot_item:
                history.add_message(ChatMessageContent(role="assistant", content=plot_item["text"]))
            else:
                print("[System] No plot generated.")

            history.add_message(result)

        except Exception as e:
            logger.error("Unhandled exception in conversation loop:")
            traceback.print_exc()
            error_message = (
                f"⚠️ Unexpected error: {type(e).__name__}: {str(e)}\nTry typing something else or restarting with 'restart'."
            )
            print("\nAssistant >", error_message, "\n")
            history.add_message(ChatMessageContent(role="assistant", content=error_message))


    save_chat_history(history, date_and_time)
    openai_cost_tracker.track_tokens(
        n_prompt_tokens=chat_completion.prompt_tokens,
        n_completion_tokens=chat_completion.completion_tokens
    )


if __name__ == '__main__':
    kernel = init_kernel()
    settings = init_settings(kernel)
    
    # Print all registered plugin functions
    for plugin_name, functions in kernel.plugins.items():
        for func_name in functions:
            print(f"{plugin_name}.{func_name}")

    
    openai_cost_tracker = OpenAICostTracker('gpt-4o-mini')
    asyncio.run(basic_agent(kernel, settings, openai_cost_tracker))
    # openai_cost_tracker.write_out_cost()
