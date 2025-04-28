import numpy as np
import polars as pl
import requests, uuid, json
from typing import List, Tuple, Optional, Any # Added Any for the loop
import os
from dotenv import load_dotenv

load_dotenv('.env')

# Add your key and endpoint
key = os.getenv("AZURE_TEXT_TRANSLATION_KEY")
endpoint = os.getenv("AZURE_TEXT_TRANSLATION_ENDPOINT")

if not key or not endpoint:
    raise ValueError("Azure Text Translation KEY and ENDPOINT must be set in .env")

location = "eastus" # You might want to make this configurable
path = "/translate"
constructed_url = endpoint + path

class Translator():
    def __init__(self, input_path: str, output_path: str):
        # Renamed input/output to input_path/output_path for clarity
        self.input_path = input_path
        self.output_path = output_path

    def create_df(self) -> pl.DataFrame:
        """Reads the input CSV file into a Polars DataFrame."""
        df = pl.read_csv(self.input_path)
        return df

    # Keep the translate_text method if you need it for single strings elsewhere,
    # but it's not used for the vectorized column translation approach below.
    # You might remove it to avoid confusion.
    # def translate_text(self, text: str, translate_to_language: List[str] = ['en']) -> Tuple[str|None, List[int]]:
    #     # ... (your existing translate_text implementation) ...
    #     pass # Placeholder

    def translate_series(self, s: pl.Series, translate_to_language: List[str] = ['en']) -> Tuple[pl.Series, pl.Series]:
        """
        Translates a Polars Series of texts using the Azure Text Translation API
        in a vectorized manner and returns two new Series: translated text and lengths.
        """
        if s.dtype != pl.String:
             print(f"Warning: Input series '{s.name}' is not of type pl.String. Attempting conversion.")
             s = s.cast(pl.String)


        # 1. Prepare data for the API call
        # Extract non-null strings and map them back to their original index
        request_data = []
        original_indices = []
        texts_list: List[Optional[str]] = s.to_list() # Get data as a Python list

        for i, text in enumerate(texts_list):
            # Check for Polars Null, Python None, or empty string if you want to skip those
            # Here, we specifically target *nulls* as per your question.
            # Empty strings "" *are* valid input for the API.
            if text is not None and text != pl.Null:
                 # Ensure it's treated as a string for the API call body
                request_data.append({'text': str(text)})
                original_indices.append(i)

        # Handle the case where there are no non-null texts to translate
        if not request_data:
             print("No non-null texts found to translate.")
             # Return two series of the same length as input, filled with nulls/defaults
             translated_texts = [None] * len(s)
             source_lengths = [[0]] * len(s) # Use [[0]] to match the List[int] type hint
             translated_lengths = [[0]] * len(s) # Use [[0]] to match the List[int] type hint
             return pl.Series("translated_text", translated_texts, dtype=pl.String), \
                    pl.Series("source_text_length", source_lengths, dtype=pl.List(pl.Int64)),\
                    pl.Series("translated_text_length", translated_lengths, dtype=pl.List(pl.Int64))


        # 2. Make the vectorized API request
        params = {
            'api-version': '3.0',
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
            request = requests.post(constructed_url, params=params, headers=headers, json=request_data)
            request.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            response = request.json()

            # 3. Process the API response and map back to original indices
            translated_texts = [None] * len(s) # Initialize with None for nulls
            source_lengths: List[List[int]] = [[0]] * len(s)
            # Initialize lengths with default list [0] for nulls/errors
            translated_lengths: List[List[int]] = [[0]] * len(s)

            # The response list order should match the request_data order
            for api_response_index, item in enumerate(response):
                original_idx = original_indices[api_response_index]
                try:
                    # Extract text and lengths from the response item
                    # Assuming translation[0] is the target language specified in 'to' list [0]
                    translated_text = item['translations'][0]['text']
                    source_text_length = item['translations'][0]['sentLen']['srcSentLen']
                    # 'sentLen' is expected to be a dictionary with 'transSentLen' as a list of ints
                    translated_length = item['translations'][0]['sentLen']['transSentLen']

                    # Store results at the correct original index
                    translated_texts[original_idx] = translated_text
                    source_lengths[original_idx] = source_text_length
                    translated_lengths[original_idx] = translated_length

                except (IndexError, KeyError, TypeError) as e:
                    # Handle cases where an individual response item is malformed
                    print(f"Warning: Error processing API response for item at API index {api_response_index} (original index {original_idx}): {e}. Setting result to None/[0]. Response item: {item}")
                    # The default None/[0] is already in place, so no explicit action needed here
                    pass # Continue processing other items

        except requests.exceptions.RequestException as e:
            print(f"Error during vectorized translation API call: {e}")
            # In case of a request error, all results will be None/[0] (initialized state)
            translated_texts = [None] * len(s)
            source_lengths = [[0]] * len(s)
            translated_lengths = [[0]] * len(s)

        # 4. Create Polars Series from the results
        translated_text_series = pl.Series("translated_text", translated_texts, dtype=pl.String) # Use pl.String for nullable strings
        source_length_series = pl.Series("source_text_length", source_lengths, dtype=pl.List(pl.Int64))
        translated_length_series = pl.Series("translated_text_length", translated_lengths, dtype=pl.List(pl.Int64))

        return translated_text_series, source_length_series, translated_length_series


    def process_translation(self, column: str) -> pl.DataFrame:
        """
        Reads the input DataFrame, translates the column,
        and adds the translated text and lengths as new columns.
        """
        df = self.create_df()

        if column not in df.columns:
            raise ValueError(f"DataFrame must contain a {column} column.")

        # print(f"Translating {column} column...")
        translated_text_series, source_length_series, translated_length_series = self.translate_series(df[column])

        # Add the new columns to the DataFrame
        df = df.with_columns([
            translated_text_series,
            source_length_series,
            translated_length_series
        ])

        # print("Translation complete. Added 'translated_text' and 'translated_text_length' columns.")
        return df
    
    def split_text(self, text: str, lengths: List[int]) -> List[str]:
        sentences = []
        start = 0
        for length in lengths:
            sentences.append(text[start:start+length])
            start += length
        return sentences
    
    def split_sentences_into_rows(self, df: pl.DataFrame, source_split_column: str, translated_split_column: str) -> pl.DataFrame:
        new_rows= []
        for row in df.head(5).iter_rows(named=True):
            sentence_index = 0
            for source_sentence, translated_sentence in zip(row[source_split_column], row[translated_split_column]):
                if source_sentence.strip():
                    new_rows.append({"id": row["id"], "sentence_index": sentence_index, "source_text": source_sentence, "translated_text": translated_sentence})
                    sentence_index += 1
        return pl.DataFrame(new_rows)


if __name__=="__main__":
    # Instantiate the Translator
    # translator_instance = Translator(input_path=dummy_input_path, output_path=dummy_output_path)
    translator_instance = Translator(input_path="./text-zh-simplified.csv", output_path="./text-zh-simplified_translated.json")

    # Process the translation
    processed_df: pl.DataFrame = translator_instance.process_translation(column= 'review')

    # processed_df.with_columns(
    #     processed_df['translated_text_length'].list.eval(pl.element().cast(pl.String)).list.join(",").alias("translated_text_length_str")
    # ).write_csv("./text-zh-simplified_translated.csv")
    
    # Apply the function to create a new column with split texts
    processed_df = processed_df.with_columns([
        pl.struct(["review", "source_text_length"]).map_elements(lambda row: translator_instance.split_text(row["review"], row["source_text_length"]), return_dtype=pl.List(pl.Utf8)).alias("source_split_texts")
    ])

    processed_df = processed_df.with_columns([
        pl.struct(["translated_text", "translated_text_length"]).map_elements(lambda row: translator_instance.split_text(row["translated_text"], row["translated_text_length"]), return_dtype=pl.List(pl.Utf8)).alias("translated_split_texts")
    ])

    final_df = translator_instance.split_sentences_into_rows(processed_df, "source_split_texts", "translated_split_texts")
    
    final_df.write_csv('text-zh-simplified_translated_split.csv')
    # print(processed_df)
    # for rows in processed_df

    # processed_df.write_parquet("./text-zh-simplified_translated.parquet")
    # print(split_text_by_lengths(texts, length))

    # print(processed_df.with_columns(pl.col("translated_text").map_elements(lambda text: split_text_by_lengths(text, processed_df["translated_text_length"])).alias("split_texts")))
