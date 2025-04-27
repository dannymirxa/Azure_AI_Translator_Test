import numpy as np
import polars as pl
import requests, uuid, json
from typing import List, Tuple, Optional, Dict, Any # Added Dict
import os
from dotenv import load_dotenv

load_dotenv('.env')

# Add your key and endpoint
key = os.getenv("AZURE_TEXT_TRANSLATION_KEY")
endpoint = os.getenv("AZURE_TEXT_TRANSLATION_ENDPOINT")

if not key or not endpoint:
    raise ValueError("Azure Text Translation KEY and ENDPOINT must be set in .env")

# Consider making location configurable if needed
location = os.getenv("AZURE_TEXT_TRANSLATION_REGION", "eastus")
path = "/translate"
constructed_url = endpoint + path

# --- Constants for Azure API ---
AZURE_API_VERSION = '3.0'
# Azure limits: Check current documentation, but typically around 100 elements and 50k chars per request
AZURE_MAX_ELEMENTS_PER_REQUEST = 100
AZURE_MAX_CHARS_PER_REQUEST = 50000 # Be mindful of this limit as well

class Translator():
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path # Output path might be less relevant for sink_*

    def scan_input(self) -> pl.LazyFrame:
        """Scans the input CSV file into a Polars LazyFrame."""
        try:
            # Use scan_csv for lazy loading
            # infer_schema_length=0 prevents reading file content just for schema inference initially
            # You might need to provide dtypes if inference fails or is slow
            lf = pl.scan_csv(self.input_path, infer_schema_length=1000)
            return lf
        except Exception as e:
            print(f"Error scanning CSV file '{self.input_path}': {e}")
            raise

    def _translate_batch_api(self, texts_batch: List[Dict[str, str]], translate_to_language: List[str]) -> Optional[List[Dict[str, Any]]]:
        """
        Helper function to make a single API call for a batch of texts.
        Returns the JSON response list or None if the request fails.
        """
        if not texts_batch:
            return [] # Return empty list if batch is empty

        params = {
            'api-version': AZURE_API_VERSION,
            'to': translate_to_language,
            'includeSentenceLength': True
        }
        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Ocp-Apim-Subscription-Region': location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

        try:
            request = requests.post(constructed_url, params=params, headers=headers, json=texts_batch)
            request.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return request.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during Azure translation API call for a batch: {e}")
            # You might want more sophisticated error handling here (e.g., retries)
            return None # Indicate failure

    def translate_text_batch(self, s: pl.Series, translate_to_language: List[str] = ['en']) -> pl.Series:
        """
        Translates a Polars Series (representing a batch) using the Azure Text Translation API.
        This function is designed to be used with `map_batches`.

        Args:
            s: A Polars Series (chunk of the original column).
            translate_to_language: List of target language codes.

        Returns:
            A Polars Series of Structs, where each struct contains:
            {'translated_text': Optional[str], 'translated_text_length': List[int]}
            The Series will have the same length as the input Series 's'.
        """
        if s.dtype != pl.String:
            # Attempt conversion, though ideally the input LF has correct types
            s = s.cast(pl.String, strict=False) # strict=False allows null propagation

        # Prepare results lists, initialized with defaults for nulls/errors
        batch_size = len(s)
        translated_texts_result = [None] * batch_size
        lengths_result: List[List[int]] = [[0]] * batch_size # Default length for null/error

        # --- Prepare data for API call(s), respecting Azure limits ---
        original_indices_map = {} # Maps API request index to original batch index
        current_request_data = []
        current_request_char_count = 0
        api_request_idx = 0

        texts_list: List[Optional[str]] = s.to_list() # Convert batch Series to list

        # Process texts for the current batch
        for original_batch_idx, text in enumerate(texts_list):
            # Skip nulls or potentially empty strings if desired
            if text is None or text == "": # Check for None or empty string
                 # Default values are already set, so just continue
                continue

            text_len = len(text)
            # Check if adding this text exceeds limits
            if (len(current_request_data) >= AZURE_MAX_ELEMENTS_PER_REQUEST or
                current_request_char_count + text_len > AZURE_MAX_CHARS_PER_REQUEST):

                # --- Send the current accumulated batch ---
                if current_request_data:
                    api_response = self._translate_batch_api(current_request_data, translate_to_language)
                    if api_response is not None:
                        # Process response and map back
                        self._process_api_response(api_response, original_indices_map, translated_texts_result, lengths_result)
                    # else: API call failed, defaults remain in results

                # Reset for the next API batch
                current_request_data = []
                original_indices_map = {}
                current_request_char_count = 0
                api_request_idx = 0
                # --- End send block ---


            # Add current text to the *next* API batch
            current_request_data.append({'text': str(text)}) # Ensure string type
            original_indices_map[api_request_idx] = original_batch_idx
            current_request_char_count += text_len
            api_request_idx += 1


        # --- Send any remaining texts after the loop ---
        if current_request_data:
            api_response = self._translate_batch_api(current_request_data, translate_to_language)
            if api_response is not None:
                self._process_api_response(api_response, original_indices_map, translated_texts_result, lengths_result)
            # else: API call failed, defaults remain

        # --- Construct the result Series of Structs ---
        # Combine the results into a list of dictionaries (structs)
        result_structs = [
            {"translated_text": txt, "translated_text_length": lng}
            for txt, lng in zip(translated_texts_result, lengths_result)
        ]

        # Create the final Series with the correct Struct dtype
        # Naming the series is not strictly necessary here as it will be named by with_columns/alias
        return pl.Series(values=result_structs, dtype=pl.Struct([
            pl.Field("translated_text", pl.String),
            pl.Field("translated_text_length", pl.List(pl.Int64))
        ]))


    def _process_api_response(self, api_response: List[Dict[str, Any]], original_indices_map: Dict[int, int], translated_texts_result: List[Optional[str]], lengths_result: List[List[int]]):
        """ Helper to process API response and populate result lists based on original indices. """
        for api_response_index, item in enumerate(api_response):
            # Find the original index in the input batch Series
            if api_response_index not in original_indices_map:
                 print(f"Warning: API response index {api_response_index} not found in original map. Skipping.")
                 continue
            original_idx = original_indices_map[api_response_index]

            try:
                translation = item.get('translations', [])[0] # Get first translation
                detected_lang_info = item.get('detectedLanguage', {}) # Get detected language info if present

                # You might want to check confidence score here: detected_lang_info.get('score', 1.0)

                translated_text = translation['text']
                # Ensure 'sentLen' and 'transSentLen' exist
                sentence_lengths = translation.get('sentLen', {}).get('transSentLen', [0]) # Default to [0] if not found

                # Store results at the correct original index
                translated_texts_result[original_idx] = translated_text
                lengths_result[original_idx] = sentence_lengths if sentence_lengths else [0] # Ensure not empty

            except (IndexError, KeyError, TypeError) as e:
                # Handle cases where an individual response item is malformed or translation failed
                print(f"Warning: Error processing API response item for original index {original_idx}: {e}. Setting result to None/[0]. Response item: {item}")
                # Defaults (None, [0]) are already set, so no explicit action needed here.

    def process_translation_lazy(self, column: str) -> pl.LazyFrame:
        """
        Creates a LazyFrame pipeline to read, translate a column in batches,
        and add the results as new columns.

        Args:
            column (str): The name of the column containing text to translate.

        Returns:
            pl.LazyFrame: A LazyFrame representing the computation plan.
                          Execute with .collect() or .sink_parquet()/.sink_csv().
        """
        lf = self.scan_input()

        if column not in lf.columns:
            # lf.columns actually triggers schema scan here if not already done
            raise ValueError(f"Input file must contain a '{column}' column. Available columns: {lf.columns}")

        print(f"Setting up lazy translation for '{column}' column...")

        # Define the return type of the map_batches function for Polars optimization
        # This must match the structure returned by translate_text_batch
        return_dtype = pl.Struct([
            pl.Field("translated_text", pl.String),
            pl.Field("translated_text_length", pl.List(pl.Int64))
        ])

        # Apply the translation function using map_batches
        lf_with_translation = lf.with_columns(
            # Apply the function to the specified column
            # The lambda creates a closure capturing `self` and `column`
            pl.col(column).map_batches(
                lambda batch_series: self.translate_text_batch(batch_series),
                return_dtype=return_dtype,
                 is_elementwise=False, # Important: function operates on whole batches
            ).alias("translation_results") # Temporary name for the struct column
        )

        # Unnest the struct column into individual columns
        lf_final = lf_with_translation.unnest("translation_results")

        # print("Lazy translation plan created. Execute with .collect() or .sink_*().")
        return lf_final

# --- Main Execution ---
if __name__=="__main__":
    # Ensure input/output paths are correct
    INPUT_CSV = "./text-zh-simplified.csv"
    # Output path for sink
    OUTPUT_PARQUET = "./text-zh-simplified_translated_lazy.parquet"
    # OUTPUT_CSV = "./text-zh-simplified_translated_lazy.csv" # Alternative

    # Instantiate the Translator
    translator_instance = Translator(input_path=INPUT_CSV, output_path=OUTPUT_PARQUET)

    # Get the LazyFrame computation plan
    processed_lf: pl.LazyFrame = translator_instance.process_translation_lazy(column='review')

    # Execute the plan and write to Parquet using sink_parquet for streaming
    print(f"Executing lazy plan and writing results to {OUTPUT_PARQUET}...")
    try:
        processed_lf.sink_parquet(
            OUTPUT_PARQUET,
            compression='zstd' # Choose a compression algorithm
        )
        # Or use sink_csv:
        # processed_lf.sink_csv(OUTPUT_CSV)

        print("Processing and writing complete.")

        # Optional: Verify output by reading a few rows
        print("\nVerifying output (first 5 rows):")
        df_head = pl.read_parquet(OUTPUT_PARQUET, n_rows=10) # Read only head for verification
        print(df_head)

    except Exception as e:
        print(f"Error during lazy execution or writing to sink: {e}")