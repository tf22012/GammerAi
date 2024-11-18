from flask import Flask, request, render_template
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration

app = Flask(__name__)

qa_model = pipeline("question-answering","distilbert/distilbert-base-cased-distilled-squad")

@app.route("/", methods=["GET", "POST"])
def home():
    tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
    model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")
    input_text = 'Fix grammatical errors in this sentence: When I grow up, I start to understand what he said is quite right.'
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=256)
    edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return render_template("home.html", input_text = input_text, edited_text = edited_text)



