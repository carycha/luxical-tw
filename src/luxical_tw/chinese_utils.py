from opencc import OpenCC
import unicodedata

class ChineseNormalizer:
    """A tool for normalizing Chinese text, supporting Traditional Chinese conversions."""

    def __init__(self, config: str = 's2twp'):
        """
        Initializes the ChineseNormalizer with the specified OpenCC configuration.
        
        Args:
            config: The OpenCC configuration to use. Defaults to 's2twp' 
                    (Simplified to Taiwan Traditional with phrases).
        """
        self.converter = OpenCC(config)

    def convert(self, text: str) -> str:
        """
        Converts the input text based on the internal configuration (OpenCC).
        
        Args:
            text: The text to convert.
        Returns:
            The converted text.
        """
        if not text:
            return ""
        return self.converter.convert(text)

    def normalize_full_to_half(self, text: str) -> str:
        """
        Converts full-width alphanumeric and symbols to half-width.
        
        Args:
            text: The text to normalize.
        Returns:
            The normalized text.
        """
        if not text:
            return ""
        # Use NFKC normalization to handle full-width characters
        return unicodedata.normalize('NFKC', text)

    def normalize(self, text: str) -> str:
        """
        Complete normalization pipeline: full-to-half width then Chinese conversion.
        
        Args:
            text: The raw text.
        Returns:
            The fully normalized text.
        """
        text = self.normalize_full_to_half(text)
        text = self.convert(text)
        return text
