# smarts-llm-annotator

Generates concise, human-readable names for SMARTS substructure filters
using an LLM, grounded with two kinds of context:

- **PubChem lookups**: for each SMARTS pattern, the smallest matching
  compounds (by molecular weight) are fetched from PubChem and their IUPAC
  names are given to the LLM as inspiration.
- **Few-shot examples**: the most structurally similar patterns (by Morgan
  fingerprint / Tanimoto similarity) from a labeled reference set
  (`data/smarts_examples.csv`) are included as naming examples.

This is a general-purpose library, meant to be installed as a dependency by
projects with their own SMARTS data (e.g. a vendor-specific label format
that needs its own preprocessing step first, or an existing database of
structure filters) rather than used standalone.

## Setup

Requires Python >=3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then fill in your API key
```

`.env` must define:

```env
API_BASE=https://chat.ai.e-infra.cz/api
API_KEY=your_api_key_here
```

## Project layout

```
smarts_llm_annotator/
  pubchem_lookup.py           PubChem SMARTS->CID and property lookups (rate-limited,
                               respects NCBI's requested off-peak window)
  prompting.py                Few-shot example selection + prompt building
  name_formatting.py          Canonical display-name formatting/cleanup
  llm_utils.py                Async batched LLM calls over a DataFrame, model discovery
  smarts_annotation_pipeline.py  Ties the above into one end-to-end pipeline
                               (see its module docstring for the step-by-step breakdown)
data/                         Reference/example CSVs used by the pipeline and its tests
tests/                        Unit tests (no network calls)
```

## Running the pipeline

```python
from smarts_llm_annotator import smarts_annotation_pipeline

out_df = await smarts_annotation_pipeline.run(
    "data/my_input.csv",  # a CSV path, or an already-loaded DataFrame
    smarts_col="smarts",
    out_path="data/output.csv",
    model_name="gpt-oss-120b",  # any model your API_BASE serves
)
```

Or from the command line:

```bash
uv run python -m smarts_llm_annotator.smarts_annotation_pipeline data/my_input.csv smarts --out-path data/output.csv --model gpt-oss-120b
```

`model_name`/`--model` can be any model name your `API_BASE` endpoint
serves. If omitted, it's resolved automatically: in an interactive
terminal you'll be prompted to choose from the models available through
your API key; otherwise the first one listed is used. To see what's
available yourself:

```python
await llm_utils.list_available_models()
```

By default, PubChem calls respect NCBI's requested off-peak window
(weekday 9pm-5am ET, or anytime on weekends). Pass `--ignore-time-window`
(CLI) or `ignore_time_window=True` (Python) to skip that, e.g. for quick
local testing.

## Tests

```bash
uv run pytest
```

## Roadmap

- **Local PubChem bulk cache**: PubChem property lookups
  (`pubchem_lookup.get_compound_properties`) currently always hit the live
  API. An earlier draft (`data_manager.py`, removed) downloaded PubChem's
  full `CID-Mass`/`CID-IUPAC` extras dumps once and served lookups from
  local files instead -- trading a large one-time download for near-zero
  marginal cost per lookup, and much less repeated load on NCBI for
  iterative/heavy use. Worth reviving as an opt-in `use_local=` mode on
  `lookup_smallest_mw_from_smarts`, fixed up (the original had a column
  name/case mismatch with the rest of the code) and with a real answer for
  cache staleness (PubChem's extras files are updated periodically).
