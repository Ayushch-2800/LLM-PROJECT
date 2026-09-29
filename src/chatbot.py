from transformers import pipeline
from src.intent_classifier import IntentClassifier
from src.multilingual import MultilingualHandler

class ContextMultilingualChatbot:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.multilingual = MultilingualHandler()
        self.generator = pipeline(
            "text-generation",
            model="gpt2",
            max_new_tokens=60,
            pad_token_id=50256
        )
        self.history = []

    def process_message(self, user_input: str, target_language: str = "en") -> dict:
        english_input = self.multilingual.translate_to_english(user_input)
        intent_data = self.intent_classifier.predict_intent(english_input)
        intent = intent_data["top_intent"]

        context = ""
        for turn in self.history[-3:]:
            u = turn["user"]
            b = turn["bot"]
            context += f"User: {u}\nBot: {b}\n"

        prompt = f"{context}User: {english_input}\nIntent: {intent}\nBot:"

        intent_responses = {
            "greeting": "Hello! How can I assist you with your project today?",
            "farewell": "Goodbye! Let me know if you need any more help.",
            "technical support": "I understand you have a technical issue. Let's troubleshoot it together.",
            "billing & pricing": "For details on pricing or subscriptions, feel free to ask.",
            "product query": "I can help answer questions regarding features and usage.",
            "general knowledge": "That's an interesting query! Let me explain."
        }

        try:
            raw_gen = self.generator(prompt, num_return_sequences=1)[0]["generated_text"]
            bot_reply = raw_gen.split("Bot:")[-1].split("\n")[0].strip()
            if not bot_reply or len(bot_reply) < 4:
                bot_reply = intent_responses.get(intent, "How else may I help you?")
        except Exception:
            bot_reply = intent_responses.get(intent, "How else may I help you?")

        final_response = self.multilingual.translate_from_english(bot_reply, target_language)
        self.history.append({"user": english_input, "bot": bot_reply, "intent": intent})

        return {
            "response": final_response,
            "intent": intent,
            "confidence": intent_data["confidence"]
        }

    def clear_memory(self):
        self.history = []
