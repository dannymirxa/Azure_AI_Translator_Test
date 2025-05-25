import polars as pl
from lingua import Language, LanguageDetectorBuilder

# languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
language = detector.detect_language_of("Smoga kedepan agar lebih ditingkatkan lagi terkait training dan pelatihan utk meningkatkan pencapaian kinerja unit")

file = "text-zh-simplified"

df = pl.scan_csv(f'./data/src/{file}.csv')

# Define a function to detect language and return the ISO code
def detect_language_iso(text: str) -> str:
    if text is None or text.strip() == "":
        return "unknown"
    lang = detector.detect_language_of(text)
    return lang.iso_code_639_1.name.lower() if lang else "unknown"

def df_language_verified(df: pl.LazyFrame) -> pl.LazyFrame:
    # Load the LazyFrame and add the language detection column
    lf = df.with_columns([
            pl.col("comments").map_elements(detect_language_iso, return_dtype=pl.String).alias("comments_language_id")
        ]).with_columns([
            pl.lit(True).alias("verified")
        ])

    return lf

df_language_verified(df).sink_csv(f'./data/src/{file}_detected.csv')