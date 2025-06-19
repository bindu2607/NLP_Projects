from main import translate_pipeline

def test_translate_en_to_zh():
    class DummyFile:
        def read(self):
            with open("tests/assets/sample.wav", "rb") as f:
                return f.read()

    dummy_audio = DummyFile()
    result = translate_pipeline(dummy_audio, "en", "zh")
    translated_text = result.get("translation", "").replace("\n", "").strip()

    reference_translation = (
        "陈啤酒的霉味挥之不去。"
        "需要加热才能散发出气味。"
        "冷水浴能恢复健康和活力。"
        "咸腌黄瓜配火腿味道很好。"
        "牧羊人玉米饼是我最喜欢的食物。"
        "热十字面包是一种令人振奋的食物。"
    )

    print("\n[EN → ZH] Translated text:", translated_text)
    print("[EN → ZH] Reference text :", reference_translation)

    assert isinstance(translated_text, str), "Translated text should be a string"
    assert translated_text, "Translated text should not be empty"
