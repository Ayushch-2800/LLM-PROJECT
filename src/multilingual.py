from deep_translator import GoogleTranslator

class MultilingualHandler:
    def translate_to_english(self, text: str) -> str:
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except Exception:
            return text

    def translate_from_english(self, text: str, target_lang: str) -> str:
        if target_lang == "en":
            return text
        try:
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
        except Exception:
            return text
