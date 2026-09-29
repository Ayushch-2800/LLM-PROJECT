from transformers import pipeline

class IntentClassifier:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-distilroberta-base"
        )
        self.candidate_labels = [
            "greeting",
            "technical support",
            "product query",
            "general knowledge",
            "billing & pricing",
            "farewell"
        ]

    def predict_intent(self, text: str) -> dict:
        res = self.classifier(text, candidate_labels=self.candidate_labels)
        return {
            "top_intent": res["labels"][0],
            "confidence": round(float(res["scores"][0]), 4)
        }
