from transformers import pipeline 
my_pipeline = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print(my_pipeline("I love this movie!"))
