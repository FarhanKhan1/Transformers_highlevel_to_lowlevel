from transformers import pipeline
import numpy as np

print("Loading model...")
classifier = pipeline("zero-shot-classification")
print("Ready!")

# classifier(
#         "This is a course about the Transformers library",
#         candidate_labels=["education", "politics", "business"],
#     )

while True:
    text = input(["Enter text (q to quit): "])
    if text.lower() == "q":
        break
    labels = input("Enter comma separated labels like (sport,education,politics,...)").split(",")
    result = classifier(text, candidate_labels=labels)
    print(result)
    print(result["labels"][np.argmax(np.array(result["scores"]))])