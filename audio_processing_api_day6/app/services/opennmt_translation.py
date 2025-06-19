import requests

class OpenNMTTranslation:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")

    def translate(self, text: str, src_lang: str = "en", tgt_lang: str = "fr") -> str:
        # Adapt to your OpenNMT REST API format
        payload = [{"src": text}]
        url = f"{self.api_url}/translate"
        response = requests.post(url, json=payload)
        if response.ok:
            # The response format may vary; adjust as needed
            result = response.json()
            if isinstance(result, list) and "tgt" in result[0]:
                return result[0]["tgt"]
            elif "result" in result:
                return result["result"]
            else:
                return str(result)
        else:
            raise RuntimeError(f"OpenNMT API error: {response.text}")
