## Setup

Create a `.env` file in the project root:

```env
API_BASE=https://chat.ai.e-infra.cz/api
API_KEY=your_api_key_here
```

Or copy the example:

```bash
cp .env.example .env
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
