import numpy as np
import polars as pl
import requests, uuid, json
from typing import List, Tuple, Optional, Any
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
    def __init__(self, input_path: str):
        self.input_path = input_path

    def create_df(self) -> pl.DataFrame:
        """Reads the input CSV file into a Polars DataFrame."""
        df = pl.read_csv(self.input_path)
        return df

    def translate_series(self, s: pl.Series, translate_to_language: List[str] = ['en']) -> Tuple[pl.Series, pl.Series]:
        """
        Translates a Polars Series of texts using the Azure Text Translation API
        in a vectorized manner and returns two new Series: translated text and lengths.
        """
        if s.dtype != pl.String:
             print(f"Warning: Input series '{s.name}' is not of type pl.String. Attempting conversion.")
             s = s.cast(pl.String)

        request_data = []
        original_indices = []
        texts_list: List[Optional[str]] = s.to_list()

        for i, text in enumerate(texts_list):
            if text is not None and text != pl.Null:
                request_data.append({'text': str(text)})
                original_indices.append(i)

        if not request_data:
             print("No non-null texts found to translate.")
             translated_texts = [None] * len(s)
             source_lengths = [[0]] * len(s)
             translated_lengths = [[0]] * len(s)
             return pl.Series("translated_text", translated_texts, dtype=pl.String), \
                    pl.Series("source_text_length", source_lengths, dtype=pl.List(pl.Int64)),\
                    pl.Series("translated_text_length", translated_lengths, dtype=pl.List(pl.Int64))

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
            request.raise_for_status()
            response = request.json()

            translated_texts = [None] * len(s)
            source_lengths: List[List[int]] = [[0]] * len(s)
            translated_lengths: List[List[int]] = [[0]] * len(s)

            for api_response_index, item in enumerate(response):
                original_idx = original_indices[api_response_index]
                try:
                    translated_text = item['translations'][0]['text']
                    source_text_length = item['translations'][0]['sentLen']['srcSentLen']
                    translated_length = item['translations'][0]['sentLen']['transSentLen']

                    translated_texts[original_idx] = translated_text
                    source_lengths[original_idx] = source_text_length
                    translated_lengths[original_idx] = translated_length

                except (IndexError, KeyError, TypeError) as e:
                    print(f"Warning: Error processing API response for item at API index {api_response_index} (original index {original_idx}): {e}. Setting result to None/[0]. Response item: {item}")
                    pass

        except requests.exceptions.RequestException as e:
            print(f"Error during vectorized translation API call: {e}")
            translated_texts = [None] * len(s)
            source_lengths = [[0]] * len(s)
            translated_lengths = [[0]] * len(s)

        translated_text_series = pl.Series("translated_text", translated_texts, dtype=pl.String)
        source_length_series = pl.Series("source_text_length", source_lengths, dtype=pl.List(pl.Int64))
        translated_length_series = pl.Series("translated_text_length", translated_lengths, dtype=pl.List(pl.Int64))

        return translated_text_series, source_length_series, translated_length_series

    def process_translation(self, column: str) -> pl.DataFrame:
        df = self.create_df()

        if column not in df.columns:
            raise ValueError(f"DataFrame must contain a {column} column.")

        translated_text_series, source_length_series, translated_length_series = self.translate_series(df[column])

        df = df.with_columns([
            translated_text_series,
            source_length_series,
            translated_length_series
        ])

        return df
    
    def split_text(self, text: str, lengths: List[int]) -> List[str]:
        sentences = []
        start = 0
        for i, length in enumerate(lengths):
            end = min(start + length, len(text))
            if i < len(lengths) - 1:
                while end > start and text[end-1] != ' ':
                    end -= 1
                if end == start:
                    end = min(start + length, len(text))
            else:
                end = len(text)
            sentences.append(text[start:end].strip())
            start = end
        return sentences
    
    def split_sentences_into_rows(self, df: pl.DataFrame, source_split_column: str, translated_split_column: str) -> pl.DataFrame:
        new_rows= []
        for row in df.iter_rows(named=True):
            sentence_index = 0
            for source_sentence, translated_sentence in zip(row[source_split_column], row[translated_split_column]):
                if source_sentence.strip():
                    new_rows.append({"respondent_id": row["respondent id"], "sentence_index": sentence_index, "source_text": source_sentence, "translated_text": translated_sentence})
                    sentence_index += 1
        return pl.DataFrame(new_rows)


if __name__=="__main__":
    file = "Infinitas SEP 2023- text comments"

    translator_instance = Translator(input_path=f"./data/src/{file}.csv")

    processed_df: pl.DataFrame = translator_instance.process_translation(column= 'comments')

    processed_df = processed_df.with_columns([
                        pl.struct(["comments", "source_text_length"]).map_elements(lambda row: translator_instance.split_text(row["comments"], row["source_text_length"]), return_dtype=pl.List(pl.Utf8)).alias("source_split_texts")
                    ]).with_columns([
                        pl.struct(["translated_text", "translated_text_length"]).map_elements(lambda row: translator_instance.split_text(row["translated_text"], row["translated_text_length"]), return_dtype=pl.List(pl.Utf8)).alias("translated_split_texts")
                    ])

    processed_df.write_parquet(f"./data/prod/{file}-simplified_translated.parquet")

    final_df = translator_instance.split_sentences_into_rows(processed_df, "source_split_texts", "translated_split_texts")
    
    final_df.write_parquet(f"./data/prod/{file}-simplified_translated_split.parquet")
