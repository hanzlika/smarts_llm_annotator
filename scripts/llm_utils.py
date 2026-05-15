import asyncio
import logging
import os

import pandas as pd
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from dotenv import load_dotenv

from typing import Hashable

logger = logging.getLogger(__name__)

# Load variables from .env
load_dotenv()

API_BASE = os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")

# Initialize async OpenAI client
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)

async def _run_with_index(idx: Hashable, sem: asyncio.Semaphore, model_name: str, prompt: str, instructions: str = None):
    async with sem:
        model_used, answer = await run_model(model_name, prompt, instructions)
        return idx, model_used, answer


async def run_model(model_name: str, prompt: str, instructions: str = None):
    """
    Runs the model with the given prompt and optional system instructions.
    Returns: (model_name, answer or error message)
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

        answer = response.choices[0].message.content.strip()
        return model_name, answer

    except Exception as e:
        logger.exception("Error running model")
        return model_name, f"Error: {e}"

async def run_llm_on_dataframe(
    df: pd.DataFrame,
    model_name: str = 'deepseek-r1',
    prompt_column: str = None,
    base_instructions: str = None,
    max_concurrency: int = 8,
) -> pd.DataFrame:
    """
    Runs LLM calls concurrently over a DataFrame.

    Args:
        df : DataFrame containing input data.
        model_name : model to use (e.g. "gpt-4o-mini").
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
        orig_idx, model_used, answer = await fut
        status = "ok" if not answer.startswith("Error:") else "error"
    
        # use the original DataFrame index
        out_df.at[orig_idx, "llm_output"] = answer
        out_df.at[orig_idx, "llm_status"] = status
    
        tqdm.write(f"Input: {df.loc[orig_idx, prompt_column]}")
        tqdm.write(f"Response: {answer}\n")

    return out_df
