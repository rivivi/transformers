from transformers import pipeline

# Load sentiment analysis pipeline
analyzer = pipeline("sentiment-analysis")

# Example text
texts = [
    "I love using Hugging Face Transformers, it's amazing!",
    "This app is okay, but it could be faster.",
    "I really hate the traffic today."
]

# Analyze each text
for t in texts:
    result = analyzer(t)[0]
    print(f"Text: {t}\n→ Sentiment: {result['label']} (Score: {result['score']:.2f})\n")
