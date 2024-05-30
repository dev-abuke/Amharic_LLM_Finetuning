import re
import pandas as pd

class DataCleaner:
    def __init__(self):
        print("DataCleaner object created")

    def normalize_char_level_missmatch(self, input_token):
        """
        Normalizes Amharic characters in the input token to their standard form.

        This function uses regular expressions to substitute various characters
        in the input token with their standardized forms. The regular expressions
        were obtained from the Unicode Standard and the Ethiopian Language Authority.

        Parameters:
            input_token (str): The Amharic token to be normalized.

        Returns:
            str: The normalized Amharic token.
        """
        replacements = [
            # labialized characters
            (re.compile(r'(ሉ[ዋአ])'), 'ሏ'),
            (re.compile(r'(ሙ[ዋአ])'), 'ሟ'),
            (re.compile(r'(ቱ[ዋአ])'), 'ቷ'),
            (re.compile(r'(ሩ[ዋአ])'), 'ሯ'),
            (re.compile(r'(ሱ[ዋአ])'), 'ሷ'),
            (re.compile(r'(ሹ[ዋአ])'), 'ሿ'),
            (re.compile(r'(ቁ[ዋአ])'), 'ቋ'),
            (re.compile(r'(ቡ[ዋአ])'), 'ቧ'),
            (re.compile(r'(ቹ[ዋአ])'), 'ቿ'),
            (re.compile(r'(ሁ[ዋአ])'), 'ኋ'),
            (re.compile(r'(ኑ[ዋአ])'), 'ኗ'),
            (re.compile(r'(ኙ[ዋአ])'), 'ኟ'),
            (re.compile(r'(ኩ[ዋአ])'), 'ኳ'),
            (re.compile(r'(ዙ[ዋአ])'), 'ዟ'),
            (re.compile(r'(ጉ[ዋአ])'), 'ጓ'),
            (re.compile(r'(ደ[ዋአ])'), 'ዷ'),
            (re.compile(r'(ጡ[ዋአ])'), 'ጧ'),
            (re.compile(r'(ጩ[ዋአ])'), 'ጯ'),
            (re.compile(r'(ጹ[ዋአ])'), 'ጿ'),
            (re.compile(r'(ፉ[ዋአ])'), 'ፏ'),
            # other characters
            (re.compile(r'[ሃኅኃሐሓኻ]'), 'ሀ'),
            (re.compile(r'[ሑኁዅ]'), 'ሁ'),
            (re.compile(r'[ኂሒኺ]'), 'ሂ'),
            (re.compile(r'[ሔዄ]'), 'ሄ'),
            (re.compile(r'[ሕኅ]'), 'ህ'),
            (re.compile(r'[ኆሖኾ]'), 'ሆ'),
            (re.compile(r'[ሠ]'), 'ሰ'),
            (re.compile(r'[ሡ]'), 'ሱ'),
            (re.compile(r'[ሢ]'), 'ሲ'),
            (re.compile(r'[ሣ]'), 'ሳ'),
            (re.compile(r'[ሤ]'), 'ሴ'),
            (re.compile(r'[ሥ]'), 'ስ'),
            (re.compile(r'[ሦ]'), 'ሶ'),
            (re.compile(r'[ዓኣዐ]'), 'አ'),
            (re.compile(r'[ዑ]'), 'ኡ'),
            (re.compile(r'[ዒ]'), 'ኢ'),
            (re.compile(r'[ዔ]'), 'ኤ'),
            (re.compile(r'[ዕ]'), 'እ'),
            (re.compile(r'[ዖ]'), 'ኦ'),
            (re.compile(r'[ጸ]'), 'ፀ'),
            (re.compile(r'[ጹ]'), 'ፁ'),
            (re.compile(r'[ጺ]'), 'ፂ'),
            (re.compile(r'[ጻ]'), 'ፃ'),
            (re.compile(r'[ጼ]'), 'ፄ'),
            (re.compile(r'[ጽ]'), 'ፅ'),
            (re.compile(r'[ጾ]'), 'ፆ'),
            # standardize ቊ and ኵ
            (re.compile(r'[ቊ]'), 'ቁ'),
            (re.compile(r'[ኵ]'), 'ኩ'),
        ]
        for pattern, replacement in replacements:
            input_token = pattern.sub(replacement, input_token)
        return input_token
    
    def remove_punc_and_special_chars(self, text):
        """
        Removes punctuation and special characters from the input text.

        Parameters:
            text (str): The input text.

        Returns:
            str: The text with all punctuation and special characters removed.
        """
        if text is None:
            raise ValueError("Input text cannot be None")
        try:
            normalized_text = re.sub('[\!\@\#\$\%\^\«\»\&\*\(\)\…\[\]\{\}\;\“\”\›\’\‘\"\'\:\,\.\‹\/\<\>\?\\\\|\`\´\~\-\=\+\፡\።\፤\;\፦\፥\፧\፨\፠\፣]', '',text)
            return normalized_text
        except Exception as e:
            raise ValueError(
                "An error occurred while removing punctuation and special characters from the input text. Exception: {}".format(e)) from e
    
    def remove_ascii_and_numbers(self, text_input):
        """
        Removes all ASCII characters, Arabic and Amharic numbers from the input text.

        Parameters:
            text_input (str): The input text.

        Returns:
            str: The text with all ASCII characters, Arabic and Amharic numbers removed.
        """
        if text_input is None:
            raise ValueError("Input text cannot be None")
        try:
            rm_num_and_ascii = re.sub('[A-Za-z0-9]', '', text_input)
            # remove all Amharic numeric characters
            return re.sub('[\'\u1369-\u137C\']+', '', rm_num_and_ascii)
        except Exception as e:
            raise ValueError(
                "An error occurred while removing ASCII characters and numbers from the input text. Exception: {}".format(e)) from e

    def remove_newline_and_extra_space(self, text):
        # Remove newline characters and extra spaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def convert_time_publish(time_str):
        if 'hours ago' in time_str:
            hours = int(re.search(r'\d+', time_str).group())
            return pd.Timestamp.now() - pd.Timedelta(hours=hours)
        return pd.NaT
    
    def convert_to_datetime(self, date_published):
        return pd.to_datetime(date_published, format='%Y/%m/%d %H:%M %Z', errors='coerce')
    
    def handle_missing_values(self, df: pd.DataFrame):
        # Handle missing values (example: filling missing with 'Unknown')
        df.fillna('Unknown', inplace=True)
    
    def export_to_csv(self, df: pd.DataFrame, file_name: str):
        df.to_csv(file_name, index=False)