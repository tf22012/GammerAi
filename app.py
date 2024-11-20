from flask import Flask, request, render_template
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# model used: https://huggingface.co/grammarly/coedit-large

tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method =="GET":
        return render_template("home.html")
    else:
        # Retrieve form input text:
        input_text = request.form["input_text"]

        # Tokenize the input text to prepare it for the model
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        # Use the model to generate an output based on the input tokens
        outputs = model.generate(input_ids, max_length=256)

        # Decode the model's output back into human-readable text
        edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return render_template("home.html", input_text = input_text, edited_text = edited_text)



