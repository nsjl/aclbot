# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

import sys
import os
import asyncio
import logging
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
from plot_from_results import plot_from_cypher_result

import streamlit as st

from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions.kernel_arguments import KernelArguments

# ---- import existing agent logic ----
from test_agent import (
    init_kernel,
    init_settings,
    extract_cypher,
    format_cypher_result,
    MAX_ASSISTANT_CHARS,
)
from paper_service import PaperSearchService
from plotter import plot


# ============================================================
# Streamlit setup
# ============================================================

st.set_page_config(
    page_title="ACLBot",
    page_icon="📚",
    layout="centered",
)

col1, col2 = st.columns([1,6])

with col1:
    st.image("logo.png", width=80)

with col2:
    st.title("ACLBot")

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


# ============================================================
# Session-state initialization
# ============================================================

if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)

    st.session_state.kernel = init_kernel()
    st.session_state.settings = init_settings(st.session_state.kernel)
    st.session_state.history = ChatHistory()
    st.session_state.started = False

    print("[System] Kernel initialized.", file=sys.stdout)


loop = st.session_state.loop
kernel = st.session_state.kernel
settings = st.session_state.settings
history: ChatHistory = st.session_state.history


# ============================================================
# One-time KG initialization (async-safe)
# ============================================================


if not st.session_state.started:

    with st.chat_message("assistant"):
        st.markdown("🔄 Initializing ACLBot knowledge graph…")

    async def _init_kg():
        return await kernel.invoke(
            plugin_name="paper_search",
            function_name="get_kg_instructions",
            arguments=KernelArguments(),
        )

    try:
        kg_result = loop.run_until_complete(_init_kg())
    except Exception as e:
        st.error(f"Failed to initialize knowledge graph: {e}")
        raise

    print(str(kg_result), file=sys.stdout)

    # history.add_message({
    #     "role": "assistant",
    #     "content": str(kg_result),
    # })

    history.add_message(
        ChatMessageContent(role="system", content=str(kg_result))
    )

    initial_message = PaperSearchService.get_initial_assistant_message()
    history.add_message(
        ChatMessageContent(role="assistant", content=initial_message)
    )

    st.session_state.started = True

    st.rerun()

# ============================================================
# Render chat history
# ============================================================

for msg in history.messages:
    if not hasattr(msg, "content"):
        continue
    if msg.role not in ("user", "assistant"):
        continue

    with st.chat_message(msg.role):
        st.markdown(msg.content)


# ============================================================
# User input
# ============================================================


user_input = st.chat_input("Ask about papers, trends, or topics…")

if user_input:

    # render user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    history.add_user_message(user_input)


    # ---- truncate history if needed ----
    for msg in history.messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            if len(msg.content) > MAX_ASSISTANT_CHARS:
                msg.content = msg.content[:MAX_ASSISTANT_CHARS] + "\n\n[Truncated early]"

    # ========================================================
    # Main agent step (async task)
    # ========================================================

    async def _agent_step():
        chat_completion: OpenAIChatCompletion = kernel.get_service(
            type=OpenAIChatCompletion
        )
        return (await chat_completion.get_chat_message_contents(
            chat_history=history,
            settings=settings,
            kernel=kernel,
            arguments=KernelArguments(input="Generate a Cypher query."),
        ))[0]

    agent_task = loop.create_task(_agent_step())

    while not agent_task.done():
        loop.run_until_complete(asyncio.sleep(0.01))

    try:
        result = agent_task.result()

        if len(result.content) > MAX_ASSISTANT_CHARS:
            logger.warning("Assistant response too long. Truncating.")
            result.content = result.content[:MAX_ASSISTANT_CHARS] + "\n\n[Truncated]"

        print("\nAssistant >", result.content, "\n", file=sys.stdout)

        # history.add_message(result)
        # with st.chat_message("assistant"):
        #     st.markdown(result.content)

        # add to history but DO NOT show yet
        history.add_message(result)

        cypher = extract_cypher(result.content)

        if not cypher:
            # normal conversational reply
            with st.chat_message("assistant"):
                st.markdown(result.content)

        # ====================================================
        # Cypher extraction + execution
        # ====================================================

        cypher = extract_cypher(result.content)
        print(cypher, file=sys.stdout)

        if cypher:
            print("[System] Extracted Cypher query:\n", cypher, file=sys.stdout)

            async def _run_cypher():
                return await kernel.invoke(
                    plugin_name="paper_search",
                    function_name="run_cypher_query",
                    arguments=KernelArguments(cypher_query=cypher),
                )

            cypher_task = loop.create_task(_run_cypher())
            while not cypher_task.done():
                loop.run_until_complete(asyncio.sleep(0.01))

            cypher_result = cypher_task.result()

            plot_specs = plot_from_cypher_result(
                user_input=user_input,
                cypher=cypher,
                result=cypher_result
            )

            # Render dynamic plots (from structured results)
            if plot_specs:
                with st.chat_message("assistant"):
                    for p in plot_specs:
                        fig, ax = plt.subplots()

                        if p["type"] == "bar":
                            ax.bar(p["x"], p["y"])
                            # rotate x labels if needed
                            ax.set_xticks(range(len(p["x"])))
                            ax.set_xticklabels(p["x"], rotation=45, ha="right")
                        elif p["type"] == "line":
                            ax.plot(p["x"], p["y"], marker="o")

                        ax.set_title(p["title"])
                        ax.set_xlabel(p["x_label"])
                        ax.set_ylabel(p["y_label"])
                        st.pyplot(fig)

            else:
                print("[System] No dynamic plot generated from Cypher results.", file=sys.stdout)


            summary_prompt = f"""Here are the query results:

{format_cypher_result(cypher_result)}

Please summarize them in a human-readable way."""
            history.add_message(
                ChatMessageContent(role="user", content=summary_prompt)
            )

            async def _summarize():
                chat_completion: OpenAIChatCompletion = kernel.get_service(
                    type=OpenAIChatCompletion
                )
                return (await chat_completion.get_chat_message_contents(
                    chat_history=history,
                    settings=settings,
                    kernel=kernel,
                    arguments=KernelArguments(),
                ))[0]

            summary_task = loop.create_task(_summarize())
            while not summary_task.done():
                loop.run_until_complete(asyncio.sleep(0.01))

            summary = summary_task.result()
            print("\nAssistant >", summary.content, "\n", file=sys.stdout)

            history.add_message(summary)

            with st.chat_message("assistant"):
                st.markdown(summary.content)

        # ====================================================
        # Plot handling
        # ====================================================

        plot_item = plot(user_input, result.content)

        if plot_item:
            print(f"[System] Rendering saved plot: {plot_item}", file=sys.stdout)

            with st.chat_message("assistant"):
                # plot_item may store the path under "text" or "path"
                img_path = plot_item.get("path") or plot_item.get("text")

                if img_path and os.path.exists(img_path):
                    st.image(img_path, width="stretch")
                else:
                    print("[System] Plot path missing or invalid.", file=sys.stdout)
        else:
            print("[System] No legacy plot generated.", file=sys.stdout)

        # st.rerun()

    except Exception as e:
        logger.error("Unhandled exception in Streamlit agent loop:")
        traceback.print_exc()

        error_message = f"⚠️ Unexpected error: {type(e).__name__}: {str(e)}"
        print(error_message, file=sys.stdout)

        history.add_message(
            ChatMessageContent(role="assistant", content=error_message)
        )

        with st.chat_message("assistant"):
            st.markdown(error_message)