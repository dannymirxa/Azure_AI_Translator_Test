import numpy as np
import polars as pl
import requests, uuid, json, os, time
from typing import List, Tuple, Optional, Any
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv('.env')

# Load your key and endpoint from the environment.
key = os.getenv("AZURE_TEXT_TRANSLATION_KEY")
endpoint = os.getenv("AZURE_TEXT_TRANSLATION_ENDPOINT")
if not key or not endpoint:
    raise ValueError("Azure Text Translation KEY and ENDPOINT must be set in .env")

location = "eastus"  # You might want to make this configurable.
path = "/translate"
constructed_url = endpoint + path

class Translator:
    def __init__(self, input_path: str, output_path: str, mini_batch_size: int = 20):
        self.input_path = input_path
        self.output_path = output_path
        self.mini_batch_size = mini_batch_size  # Number of rows per mini-batch

    def translate_series(self, s: pl.Series, translate_to_language: List[str] = ['en']) -> Tuple[pl.Series, pl.Series]:
        """
        Translates a Polars Series of texts using the Azure Text Translation API.
        Returns two Series: one for the translated text and one for the sentence lengths.
        """
        # Ensure the series is of string type.
        if s.dtype != pl.String:
            print(f"Warning: Input series '{s.name}' is not of type pl.String. Attempting conversion.")
            s = s.cast(pl.String)

        # Prepare data for the API call by filtering out null values.
        request_data = []
        original_indices = []
        texts_list: List[Optional[str]] = s.to_list()
        for i, text in enumerate(texts_list):
            if text is not None and text != pl.Null:
                request_data.append({"text": str(text)})
                original_indices.append(i)

        # Handle the case of no valid texts.
        if not request_data:
            print("No non-null texts found to translate.")
            translated_texts = [None] * len(s)
            lengths = [[0]] * len(s)
            return (pl.Series("translated_text", translated_texts, dtype=pl.String),
                    pl.Series("translated_text_length", lengths, dtype=pl.List(pl.Int64)))

        params = {
            "api-version": "3.0",
            "to": translate_to_language,
            "includeSentenceLength": True
        }
        headers = {
            "Ocp-Apim-Subscription-Key": key,
            "Ocp-Apim-Subscription-Region": location,
            "Content-type": "application/json",
            "X-ClientTraceId": str(uuid.uuid4())
        }

        try:
            response = requests.post(constructed_url, params=params, headers=headers, json=request_data)
            response.raise_for_status()
            response_json = response.json()

            # Initialize output lists with defaults.
            translated_texts = [None] * len(s)
            lengths: List[List[int]] = [[0]] * len(s)
            # Process the API response, mapping back to the original indices.
            for api_response_index, item in enumerate(response_json):
                original_idx = original_indices[api_response_index]
                try:
                    translated_text = item["translations"][0]["text"]
                    translated_length = item["translations"][0]["sentLen"]["transSentLen"]
                    translated_texts[original_idx] = translated_text
                    lengths[original_idx] = translated_length
                except (IndexError, KeyError, TypeError) as e:
                    print(f"Warning: Error processing API response for item at API index {api_response_index} "
                          f"(original index {original_idx}): {e}. Response item: {item}")
            # End of API response processing.
        except requests.exceptions.RequestException as e:
            print(f"Error during the API call: {e}")
            translated_texts = [None] * len(s)
            lengths = [[0]] * len(s)

        translated_text_series = pl.Series("translated_text", translated_texts, dtype=pl.String)
        translated_length_series = pl.Series("translated_text_length", lengths, dtype=pl.List(pl.Int64))
        return translated_text_series, translated_length_series

    def process_translation_lazy(self, column: str) -> pl.DataFrame:
        """
        Processes the translation using a LazyFrame.
        Since there is no mini-batch size control in map_batches, we explicitly slice
        the LazyFrame into mini-batches to throttle API calls.
        """
        # Create a LazyFrame by lazily scanning the CSV.
        lf = pl.scan_csv(self.input_path)

        # First, determine total row count without fully materializing data.
        total_rows = lf.select(pl.len()).collect().item()

        # Define the new schema (your CSV might contain other columns; adjust as needed).
        new_schema = {
            "id": pl.Int64,
            "review": pl.Utf8,
            "translated_text": pl.Utf8,
            "translated_text_length": pl.List(pl.Int64)
        }

        output_dfs = []
        # For each mini-batch, slice and process.
        for start in tqdm(range(0, total_rows, self.mini_batch_size)):
            # Slice the LazyFrame for a mini-batch.
            batch_lf = lf.slice(start, self.mini_batch_size)

            # Define a batch function.
            def translate_batch(df: pl.DataFrame) -> pl.DataFrame:
                if column not in df.columns:
                    raise ValueError(f"DataFrame must contain a '{column}' column.")
                # Sleep a bit to throttle API calls.
                time.sleep(0.5)
                tx_series, len_series = self.translate_series(df[column])
                return df.with_columns([
                    tx_series.alias("translated_text"),
                    len_series.alias("translated_text_length")
                ])

            # Apply the batch function using map_batches (without a batch_size parameter).
            batch_df = batch_lf.map_batches(translate_batch, schema=new_schema).collect()
            output_dfs.append(batch_df)

        # Concatenate all the mini-batches.
        final_df = pl.concat(output_dfs)
        return final_df

if __name__ == "__main__":
    translator_instance = Translator(
        input_path="./text-zh-large.csv",
        output_path="./text-zh-large_translated.parquet",
        mini_batch_size=10  # Set your desired mini-batch size here.
    )

    # Process the translation for the 'review' column using our explicit mini-batch approach.
    processed_df: pl.DataFrame = translator_instance.process_translation_lazy(column="review")
    processed_df.write_parquet(translator_instance.output_path)
