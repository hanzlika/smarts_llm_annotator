# Assuming that the user had preprocessed the data into a dataframe with a column of valid smarts
import pandas as pd
from scripts import pubchem_lookup, get_pubchem_compound_data, llm_utils
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from tqdm import tqdm 
import ast

ROOT_DIR = Path(__file__).resolve().parents[0]
DATA_DIR = ROOT_DIR / "data"
COMPOUNDS_PROPERTIES_PATH = DATA_DIR / 'CIDs_to_mass_iupac.csv'
SMARTS_EXAMPLES_PATH = DATA_DIR / 'smarts_examples.csv'
DEFAULT_OUT_PATH = DATA_DIR / 'output.csv'

def run(in_path, smarts_col, out_path=DEFAULT_OUT_PATH, use_local=True, ignore_time_window=False):
    # ensure local data is downloaded and ready if we're going to use it
    if use_local:
        get_pubchem_compound_data.ensure_file_present()
        
    df = pd.read_csv(in_path)[[smarts_col]].dropna()
    
    #1 get pubchem matching of smarts
    df = pubchem_lookup.parallel_cid_lookup(
        df,
        smarts_col,
        limit=1000,
        max_workers=8,
        show_progress=True,
        ignore_time_window=ignore_time_window
    )
    
    # 2 get n smallest MW compounds out of that list
    all_cids = set([int(cid) for cids_list in df['cids'] for cid in eval(cids_list)])

    # a) using stored pubchem data
    if use_local:
        compounds_properties = pd.read_csv(COMPOUNDS_PROPERTIES_PATH)

        # limit to cids were interested in
        compounds_properties = compounds_properties[compounds_properties['CID'].isin(all_cids)].copy()

        df['best_IUPAC_names'] = pubchem_lookup.get_lowest_mw_iupac_names(df, compounds_properties)

    else:
        return
    # b) using pubchem api
    # not implemented yet

    #3 get n most similar SMARTS
    df['similar_examples'] = get_top_n_most_similar_smarts_description_examples_wrapper(df, 'inferred_smarts', n=10)

    #4 compile into prompts with few-shot examples
    df["prompt"] = df.apply(
        lambda row: build_prompt_safe(
            row["best_IUPAC_names"],
            row["similar_examples"],
            row[smarts_col]
        ),
        axis=1
    )

    async def run_llm_on_df_async(in_df):
        out_df =  await llm_utils.run_llm_on_dataframe(in_df, prompt_column='prompt')
        out_df[[smarts_col, 'best_IUPAC_names', 'llm_output']].to_csv(out_path, index=False)

    #5 async call of the LLM
    run_llm_on_df_async(df)


def get_top_n_most_similar_smarts_description_examples_wrapper(df, smarts_col, n=10):
    ref = pd.read_csv(SMARTS_EXAMPLES_PATH)
    
    # Initialize Morgan fingerprint generator
    morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
    
    def get_fps_from_list(smarts_list):
        fps = []
        for s in smarts_list:
            mol = Chem.MolFromSmarts(s)
            
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                    fp = morgan_gen.GetFingerprint(mol)
                    fps.append(fp)
                except ValueError:
                    fps.append(None)
                    print(f"Fingerprint generation failed for SMARTS: {s}")
            else:
                fps.append(None)
                print(f"Invalid SMARTS: {s}")
        return fps

    ref['fps'] = get_fps_from_list(ref['smarts'].to_list())
    ref.dropna(inplace=True)
    smarts_fps = get_fps_from_list(df[smarts_col].to_list())

    smarts_examples = []
    for query_smarts in tqdm(smarts_fps, desc="Finding most similar smarts examples"):
        examples, are_random = get_top_n_most_similar_smarts_description_examples(query_smarts, ref, n=n)
        smarts_examples.append(examples)

    return smarts_examples


def get_top_n_most_similar_smarts_description_examples(query_fp, ref_df, n):
    # Compute similarity for each fingerprint
    if query_fp is None:
        return list(ref_df.sample(2 * n)
                          [['smarts', 'cleaned_description']]
                          .itertuples(index=False, name=None)), True
    similarities = []
    for idx, row in ref_df.iterrows():
        fp2 = row['fps']
        if fp2 is not None:
            sim = DataStructs.TanimotoSimilarity(query_fp, fp2)
            similarities.append(((row['smarts'], row['cleaned_description']), sim))  # store SMARTS with similarity
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top n SMARTS-description pairs
    return [s for s, sim in similarities[:n]], False


def build_prompt_safe(best_iupac_names, similar_examples, input_smarts):
    shared_base_instructions = """You are an LLM that converts SMARTS substructure filters into concise, human-readable names.
    - **Input**: A SMARTS string.
    - **Output**: A lowercase name with words separated by spaces.
    - The name should help molecular biologists/chemists instantly recognize the chemical feature (e.g., 'amide bond').
    - Return **ONLY the name** - no prefixes, suffixes, explanations, or formatting.
    """
    
    matching_compounds_example_prompt_base = "IUPAC names of smallest matching compounds (by atom count). Use for inspiration but note:\n- SMARTS patterns are often simpler than matching compounds\n- NEVER directly use compound names; identify general features instead:\n"
    
    similar_examples_prompt_base = "\nExamples of similar SMARTS-to-name conversions. Use to maintain consistent naming conventions:\n"
    # Safely parse inputs
    try:
        iupac_list = ast.literal_eval(best_iupac_names)
        examples_list = ast.literal_eval(similar_examples)
    except (SyntaxError, ValueError):
        # Fallback to empty lists on error
        iupac_list = []
        examples_list = []
    
    # Truncate long lists (e.g., max 5 items)
    matching_compounds_prompt = (
        matching_compounds_example_prompt_base + '\n'.join(iupac_list[:5]) 
        if iupac_list else ''
    )
    
    similar_examples_prompt = similar_examples_prompt_base + ''.join(
        f"Input: {smarts} -> Output: {name}\n" 
        for smarts, name in examples_list[:5]  # Truncate examples
    )
    
    return (
        shared_base_instructions + 
        matching_compounds_prompt + 
        similar_examples_prompt + 
        f"Now you fill in the final one:\nInput: {input_smarts} -> Output:"
    )