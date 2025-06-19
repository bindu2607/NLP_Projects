from transformers import pipeline

def setup_pipeline():
    """
    Initializes and returns the MarianMT translation pipeline.
    
    This function loads the pre-trained English-to-French model
    from Hugging Face.
    """
    model_name = 'Helsinki-NLP/opus-mt-en-fr'
    try:
        translator_pipeline = pipeline('translation', model=model_name)
        print(f"Pipeline for model '{model_name}' loaded successfully.")
        return translator_pipeline
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return None

