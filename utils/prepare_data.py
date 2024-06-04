from clean_data import DataCleaner
import pandas as pd

class PrepareData:
    def __init__(self):
        self.data_cleaner = DataCleaner()
        self.file_name = "./data/merged_data.csv"

    def prepare_data(self, df: pd.DataFrame):
        df.dropna(inplace=True)
        # clean_summary
        df['clean_summary'] = df['summary'].apply(self.data_cleaner.remove_symbols)
        df['clean_summary'] = df['clean_summary'].apply(self.data_cleaner.remove_emojis)
        df['clean_summary'] = df['clean_summary'].apply(self.data_cleaner.remove_hyperlinks)
        df['clean_summary'] = df['clean_summary'].apply(self.data_cleaner.remove_mentions)
        df["clean_summary"] = df["clean_summary"].apply(self.data_cleaner.remove_hashtags)
        df['clean_summary'] = df["clean_summary"].apply(self.data_cleaner.normalize_char_level_missmatch)
        df['clean_summary'] = df["clean_summary"].apply(self.data_cleaner.remove_english_characters)
        df['clean_summary'] = df["clean_summary"].apply(self.data_cleaner.remove_punc_and_special_chars)
        df['clean_summary'] = df["clean_summary"].apply(self.data_cleaner.remove_newline_and_extra_space)

        # clean_text
        df['clean_text'] = df['text'].apply(self.data_cleaner.remove_symbols)
        df['clean_text'] = df['clean_text'].apply(self.data_cleaner.remove_emojis)
        df['clean_text'] = df['clean_text'].apply(self.data_cleaner.remove_hyperlinks)
        df['clean_text'] = df['clean_text'].apply(self.data_cleaner.remove_mentions)
        df["clean_text"] = df["clean_text"].apply(self.data_cleaner.remove_hashtags)
        df['clean_text'] = df["clean_text"].apply(self.data_cleaner.normalize_char_level_missmatch)
        df['clean_text'] = df["clean_text"].apply(self.data_cleaner.remove_english_characters)
        df['clean_text'] = df["clean_text"].apply(self.data_cleaner.remove_punc_and_special_chars)

        return df