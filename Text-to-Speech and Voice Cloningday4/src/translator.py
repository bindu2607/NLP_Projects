from .pipeline_manager import setup_pipeline

# Initialize the pipeline once when the module is loaded
# This avoids reloading the model on every function call
translation_pipeline = setup_pipeline()

def translate_text(asr_output: str) -> str:
    """
    Takes an ASR output string and returns the translated text.

    Args:
        asr_output: The English text to be translated.

    Returns:
        The translated French text as a string.
    """
    if translation_pipeline is None:
        return "Error: Translation pipeline not available."

    if not asr_output or not isinstance(asr_output, str):
        return "Error: Invalid input. Please provide a non-empty string."

    try:
        result = translation_pipeline(asr_output)
        # The pipeline returns a list of dictionaries, e.g., [{'translation_text': '...'}]
        return result[0]['translation_text']
    except Exception as e:
        return f"Error during translation: {e}"

