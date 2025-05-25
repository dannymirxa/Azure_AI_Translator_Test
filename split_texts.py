import polars as pl
from typing import List

def split_text(text: str, lengths: List[int]) -> List[str]:
    sentences = []
    start = 0
    for i, length in enumerate(lengths):
        end = min(start + length, len(text))
        if i < len(lengths) - 1: # For all segments except the last
            while end > start and text[end-1] != ' ':
                end -= 1
            if end == start:
                end = min(start + length, len(text)) # Force split if no space found
        else:
            end = len(text) # For the last segment, just take the rest of the text
        sentences.append(text[start:end].strip())
        start = end
    return sentences

def split_sentences_into_rows(df: pl.DataFrame, source_split_column: str, translated_split_column: str) -> pl.DataFrame:
    new_rows= []
    for row in df.iter_rows(named=True):
        sentence_index = 0
        for source_sentence, translated_sentence in zip(row[source_split_column], row[translated_split_column]):
            if source_sentence.strip():
                new_rows.append({"respondent_id": row["respondent id"], "sentence_index": sentence_index, \
                                    "source_text": source_sentence, "translated_text": translated_sentence} )
                sentence_index += 1
    return pl.DataFrame(new_rows)