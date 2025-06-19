import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator:
    """
    Translator using Facebook's NLLB 1.3B model for English <-> Chinese translations.
    """

    def __init__(self):
        # Use GPU if available else CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model name
        self.model_name = "facebook/nllb-200-1.3B"
        
        # Load tokenizer and model to device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        # Language code map for NLLB model
        self.lang_code_map = {
            ("en", "zh"): ("eng_Latn", "zho_Hans"),
            ("zh", "en"): ("zho_Hans", "eng_Latn"),
        }

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate text from src_lang to tgt_lang.

        Args:
            text (str): Input text to translate.
            src_lang (str): Source language code ('en' or 'zh').
            tgt_lang (str): Target language code ('en' or 'zh').

        Returns:
            str: Translated text.

        Raises:
            ValueError: If language pair is unsupported.
        """
        # Check for empty or non-string input
        if not isinstance(text, str) or not text.strip():
            return ""

        # Validate language pair
        if (src_lang, tgt_lang) not in self.lang_code_map:
            raise ValueError(f"Unsupported language pair: {src_lang}-{tgt_lang}")

        # Get model language codes
        src_code, tgt_code = self.lang_code_map[(src_lang, tgt_lang)]
        self.tokenizer.src_lang = src_code

        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

        # Generate translation with no grad for efficiency
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
                max_length=512,
                num_beams=5,             # Beam search for better quality
                early_stopping=True
            )

        # Decode generated tokens to string
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return translated_text


