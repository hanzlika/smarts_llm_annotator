import asyncio
import logging
import os
import sys

import pandas as pd
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from dotenv import load_dotenv

from typing import Hashable, List, Optional

logger = logging.getLogger(__name__)

# Load variables from .env
load_dotenv()

API_BASE = os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")

if not API_BASE or not API_KEY:
    raise RuntimeError(
        "API_BASE and API_KEY must be set -- copy .env.example to .env "
        "and fill in your API key."
    )

# Initialize async OpenAI client
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)

async def list_available_models() -> List[str]:
    """Return the model IDs available through this API endpoint/key."""
    models = await client.models.list()
    return [m.id for m in models.data]


async def resolve_model_name(model_name: Optional[str] = None, interactive: bool = True) -> str:
    """
    Return model_name unchanged if given. Otherwise, query the API for
    available models: in an interactive terminal, ask the user to pick
    one; non-interactively (e.g. a notebook run with no model specified,
    or stdin isn't a terminal), default to the first model listed.
    """
    if model_name:
        return model_name

    models = await list_available_models()
    if not models:
        raise RuntimeError("No models are available through this API key.")

    if interactive and sys.stdin.isatty():
        print("No model specified. Available models:")
        for i, m in enumerate(models, start=1):
            print(f"  {i}. {m}")
        choice = input(f"Choose a model [1-{len(models)}] (default: 1, {models[0]}): ").strip()
        if choice:
            return models[int(choice) - 1]

    return models[0]


async def _run_with_index(idx: Hashable, sem: asyncio.Semaphore, model_name: str, prompt: str, instructions: str = None):
    async with sem:
        answer = await run_model(model_name, prompt, instructions)
        return idx, answer


async def run_model(model_name: str, prompt: str, instructions: str = None):
    """
    Runs the model with the given prompt and optional system instructions.
    Returns: answer or error message
    """
    try:
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.exception("Error running model")
        return f"Error: {e}"

async def run_llm_on_dataframe(
    df: pd.DataFrame,
    model_name: str,
    prompt_column: str = None,
    base_instructions: str = None,
    max_concurrency: int = 8,
) -> pd.DataFrame:
    """
    Runs LLM calls concurrently over a DataFrame.

    Args:
        df : DataFrame containing input data.
        model_name : model to use (e.g. "gpt-4o-mini"). Use resolve_model_name()
            to fill this in from the API's available models if not preset.
        prompt_column : column name containing the prompt or input text.
        base_instructions : optional instruction prefix (e.g. BASE_INSTRUCTIONS).
        max_concurrency : max concurrent async tasks.

    Returns:
        A copy of df with:
          - llm_output : model output or error
          - llm_status : "ok" or "error"
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    out_df = df.copy()
    out_df["llm_output"] = None
    out_df["llm_status"] = None

    tasks = []
    for orig_idx, row in df.iterrows():  # orig_idx is the real index
        if base_instructions and prompt_column:
            prompt = f"{base_instructions}Input: {row[prompt_column]}\nOutput: "
        elif prompt_column:
            prompt = row[prompt_column]
        else:
            raise ValueError("You must specify `prompt_column` for input data.")
    
        # pass the real index, not a 0...n counter
        tasks.append(asyncio.create_task(_run_with_index(orig_idx, semaphore, model_name, prompt, base_instructions)))

    
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM calls", colour="cyan"):
        orig_idx, answer = await fut
        status = "ok" if not answer.startswith("Error:") else "error"
    
        # use the original DataFrame index
        out_df.at[orig_idx, "llm_output"] = answer
        out_df.at[orig_idx, "llm_status"] = status
    
        tqdm.write(f"Input: {df.loc[orig_idx, prompt_column]}")
        tqdm.write(f"Response: {answer}\n")

    return out_df
