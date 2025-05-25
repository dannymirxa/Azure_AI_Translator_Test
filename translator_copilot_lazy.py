import numpy as np
import polars as pl
import requests, uuid, json, os, time
from typing import List, Tuple, Optional, Any
from dotenv import load_dotenv
from tqdm import tqdm
# from detect_language import df_language_verified
from split_texts import split_text, split_sentences_into_rows
import os

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
    def __init__(self, input_path: str, mini_batch_size: int = 100):
        self.input_path = input_path
        self.mini_batch_size = mini_batch_size  # Number of rows per mini-batch

    def translate_series(self, s: pl.Series, source_languages: pl.Series, translate_to_language: List[str] = ['en']) -> Tuple[pl.Series, pl.Series, pl.Series]:
        """
        Translates a Polars Series of texts using the Azure Text Translation API.
        Accepts a Series of source_languages to conditionally set the 'from' parameter.
        Returns three Series: translated text, source sentence lengths, and translated sentence lengths.
        """
        # Ensure the series is of string type.
        if s.dtype != pl.String:
            print(f"Warning: Input series '{s.name}' is not of type pl.String. Attempting conversion.")
            s = s.cast(pl.String)

        # Prepare data for the API call by filtering out null values.
        # Also, determine if a common 'from' language can be used for the batch.
        request_data = []
        original_indices = []
        texts_list: List[Optional[str]] = s.to_list()
        source_lang_list: List[Optional[str]] = source_languages.to_list()
        
        valid_source_langs_for_this_api_call = set()

        for i, (text, lang) in enumerate(zip(texts_list, source_lang_list)):
            if text is not None: # Polars Null becomes Python None in to_list()
                request_data.append({"text": str(text)})
                original_indices.append(i)
                if lang is not None and isinstance(lang, str) and lang.lower() != "unknown":
                    valid_source_langs_for_this_api_call.add(lang)

        # Handle the case of no valid texts.
        if not request_data:
            print("No non-null texts found to translate.")
            translated_texts = [None] * len(s)
            source_lengths = [[0]] * len(s)
            translated_lengths = [[0]] * len(s)
            return (pl.Series("translated_text", translated_texts, dtype=pl.String),\
                    pl.Series("source_text_length", source_lengths, dtype=pl.List(pl.Int64)),\
                    pl.Series("translated_text_length", translated_lengths, dtype=pl.List(pl.Int64)))

        params = {
            "api-version": "3.0",
            "to": translate_to_language,
            "includeSentenceLength": True
        }

        # If all texts to be translated in this batch share a single, known source language, set the 'from' parameter.
        if len(valid_source_langs_for_this_api_call) == 1:
            params["from"] = valid_source_langs_for_this_api_call.pop()

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
            source_lengths: List[List[int]] = [[0]] * len(s)
            translated_lengths: List[List[int]] = [[0]] * len(s)
            # Process the API response, mapping back to the original indices.
            for api_response_index, item in enumerate(response_json):
                original_idx = original_indices[api_response_index]
                try:
                    translated_text = item["translations"][0]["text"]
                    source_text_length = item['translations'][0]['sentLen']['srcSentLen']
                    translated_length = item["translations"][0]["sentLen"]["transSentLen"]
                    translated_texts[original_idx] = translated_text
                    source_lengths[original_idx] = source_text_length
                    translated_lengths[original_idx] = translated_length
                except (IndexError, KeyError, TypeError) as e:
                    print(f"Warning: Error processing API response for item at API index {api_response_index} "
                          f"(original index {original_idx}): {e}. Response item: {item}")
            # End of API response processing.
        except requests.exceptions.RequestException as e:
            print(f"Error during the API call: {e}")
            translated_texts = [None] * len(s)
            source_lengths = [[0]] * len(s)
            translated_lengths = [[0]] * len(s)

        translated_text_series = pl.Series("translated_text", translated_texts, dtype=pl.String)
        source_length_series = pl.Series("source_text_length", source_lengths, dtype=pl.List(pl.Int64))
        translated_length_series = pl.Series("translated_text_length", translated_lengths, dtype=pl.List(pl.Int64))
        return translated_text_series, source_length_series, translated_length_series

    def process_translation_lazy(self, column: str) -> pl.DataFrame:
        """
        Processes the translation using a LazyFrame.
        Since there is no mini-batch size control in map_batches, we explicitly slice
        the LazyFrame into mini-batches to throttle API calls.
        """
        # Create a LazyFrame by lazily scanning the CSV.
        # Only selects respondent id and comments columns
        lf = pl.scan_csv(self.input_path).select(["respondent id", "comments", "comments_language_id", "verified"])

        # First, determine total row count without fully materializing data.
        total_rows = lf.select(pl.len()).collect().item()

        # Define the new schema (your CSV might contain other columns; adjust as needed).
        new_schema = {
            "respondent id": pl.Int64,
            "comments": pl.Utf8,
            "comments_language_id": pl.Utf8,
            "verified": pl.Boolean,
            "translated_text": pl.Utf8,
            "source_text_length": pl.List(pl.Int64),
            "translated_text_length": pl.List(pl.Int64),
        }

        # Define a batch function.
        def translate_batch(df: pl.DataFrame) -> pl.DataFrame:
            if column not in df.columns:
                raise ValueError(f"DataFrame must contain a '{column}' column.")

            # Determine which rows need translation
            needs_translation_mask = (df["comments_language_id"] != "en") | (df["comments_language_id"] == "unknown") & (df["verified"] == True)

            # Split the DataFrame
            df_to_translate = df.filter(needs_translation_mask)
            df_no_translate = df.filter(~needs_translation_mask)

            # Translate only the necessary rows
            if df_to_translate.height > 0:
                translated_text_series, source_len_series, translated_len_series = self.translate_series(df_to_translate[column], df_to_translate["comments_language_id"], translate_to_language=['en'])
                df_to_translate = df_to_translate.with_columns([
                    translated_text_series.alias("translated_text"),
                    source_len_series.alias("source_text_length"),
                    translated_len_series.alias("translated_text_length")
                ])
            else:
                df_to_translate = df_to_translate.with_columns([
                    pl.lit(None).alias("translated_text").cast(pl.Utf8),
                    pl.lit([0]).alias("source_text_length").cast(pl.List(pl.Int64)),
                    pl.lit([0]).alias("translated_text_length").cast(pl.List(pl.Int64))
                ])

            # For rows that don't need translation, copy the original text and compute lengths
            df_no_translate = df_no_translate.with_columns([
                df_no_translate[column].alias("translated_text"),
                pl.col(column).map_elements(lambda x: [len(x)] if x else [0], return_dtype=pl.List(pl.Int64)).alias("source_text_length"),
                pl.col(column).map_elements(lambda x: [len(x)] if x else [0], return_dtype=pl.List(pl.Int64)).alias("translated_text_length")
            ])

            # Combine both parts and sort to maintain original order
            return pl.concat([df_to_translate, df_no_translate]).sort("respondent id")

        output_dfs = []
        # For each mini-batch, slice and process.
        for start in tqdm(range(0, total_rows, self.mini_batch_size)):
            batch_index = start // self.mini_batch_size
            batch_file = f"./data/temp/batch_{batch_index:03d}.parquet"

            # Skip if already processed
            if os.path.exists(batch_file):
                print(f"Batch {batch_index} already processed. Skipping.")
                continue

            # Slice the LazyFrame for a mini-batch.
            batch_lf = lf.slice(start, self.mini_batch_size)

            # Apply the batch function using map_batches (without a batch_size parameter).
            batch_df = batch_lf.map_batches(translate_batch, schema=new_schema).collect()

            # Save batch immediately
            batch_df.write_parquet(batch_file)
            output_dfs.append(batch_df)

        
        # Load all saved batches
        batch_files = sorted([f for f in os.listdir("./data/temp") if f.startswith("batch_") and f.endswith(".parquet")])
        final_df = pl.concat([pl.read_parquet(f"./data/temp/{f}") for f in batch_files])

        
        temp_dir = "./data/temp"

        for file in os.listdir(temp_dir):
            if file.endswith(".parquet"):
                os.remove(os.path.join(temp_dir, file))

        return final_df

if __name__ == "__main__":
    # file = "Infinitas SEP 2023- text comments"
    file = "text-zh-simplified_detected"

    translator_instance = Translator(
        input_path=f"./data/src/{file}.csv",
        mini_batch_size=50  # Set your desired mini-batch size here.
    )

    # Process the translation for the 'comments' column using our explicit mini-batch approach.
    processed_df =  translator_instance.process_translation_lazy(column="comments")

    processed_df = processed_df.with_columns([
                    pl.struct(["comments", "source_text_length"]).map_elements(lambda row: split_text(row["comments"], row["source_text_length"]), return_dtype=pl.List(pl.Utf8)).alias("source_split_texts")
                ]).with_columns([
                    pl.struct(["translated_text", "translated_text_length"]).map_elements(lambda row: split_text(row["translated_text"], row["translated_text_length"]), return_dtype=pl.List(pl.Utf8)).alias("translated_split_texts")
                ])

    processed_df.write_parquet(f"./data/prod/{file}-simplified_translated_lazy.parquet")

    final_df = split_sentences_into_rows(processed_df, "source_split_texts", "translated_split_texts")
    final_df.write_parquet(f"./data/prod/{file}-simplified_translated_split_lazy.parquet")