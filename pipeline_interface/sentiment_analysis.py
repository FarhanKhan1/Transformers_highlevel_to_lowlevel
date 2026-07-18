from transformers import pipeline

print("Loading model...")
classifier = pipeline("sentiment-analysis")
print("Ready!")

while True:
    text = input(["Enter text (q to quit): "])

    if text.lower() == "q":
        break

    print(classifier(text))