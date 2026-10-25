"""Live smoke test for the fallback chain. Needs real keys in .env.

    python scripts/smoke_llm.py

Runs three checks: a plain completion, a tool call, and a forced fallback where the
first model in the chain is a bogus id so the chain must skip it.
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.services.llm import FallbackLLM, LLMSettings, Message, ModelRef, ToolSpec  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def show(label: str, resp) -> None:
    trail = " -> ".join(
        f"{a.provider}:{a.model}({'ok' if a.ok else a.error_kind}, {a.latency_ms}ms)" for a in resp.attempts
    )
    print(f"\n== {label}")
    print(f"   answered by : {resp.provider}:{resp.model}  fell_back={resp.fell_back}")
    print(f"   attempts    : {trail}")
    print(f"   usage       : in={resp.usage.input_tokens} out={resp.usage.output_tokens}")
    for tc in resp.tool_calls:
        print(f"   tool call   : {tc.name}({tc.arguments})")
    if resp.text:
        print(f"   text        : {resp.text.strip()[:300]}")


def main() -> None:
    settings = LLMSettings.from_env()
    print("chain:", " -> ".join(r.key for r in settings.chain()))

    llm = FallbackLLM.from_settings(settings)

    # 1. plain completion
    show(
        "plain completion",
        llm.invoke(
            [
                Message.system("You are a concise assistant for Sri Lankan law."),
                Message.user("In one sentence, what does Article 12 of the Sri Lankan Constitution guarantee?"),
            ],
            temperature=0.0,
        ),
    )

    # 2. tool call
    tool = ToolSpec(
        name="search_statutes",
        description="Search Sri Lankan statutes and case law.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"},
                "year_from": {"type": "integer", "description": "earliest year to include"},
            },
            "required": ["query"],
        },
    )
    show(
        "tool call",
        llm.invoke(
            [Message.user("Find amendments to the Prevention of Terrorism Act since 2020. Use the search tool.")],
            tools=[tool],
            temperature=0.0,
        ),
    )

    # 3. forced fallback: bogus model first, then the real chain
    forced = FallbackLLM(
        [ModelRef(settings.provider, "this-model-does-not-exist"), *settings.chain()],
        llm.providers,
        max_retries_per_model=settings.max_retries_per_model,
        timeout_s=settings.timeout_s,
    )
    show("forced fallback (bogus model first)", forced.invoke([Message.user("Reply with the single word: ready")]))
    print("\ncooldowns now:", forced.cooldowns())

    # 4. groq directly, so the cross-provider leg is proven to work too
    if "groq" in llm.providers:
        groq_only = FallbackLLM([r for r in settings.chain() if r.provider == "groq"], llm.providers, timeout_s=settings.timeout_s)
        show("groq leg", groq_only.invoke([Message.user("Reply with the single word: ready")], tools=[tool]))


if __name__ == "__main__":
    main()
