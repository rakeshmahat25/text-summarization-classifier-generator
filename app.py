from transformers import pipeline
from flask import Flask, render_template, request


app = Flask(__name__)

# Initialize pipelines
generator = pipeline('text-generation', model='gpt2')
classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')


@app.route("/", methods=["GET", "POST"])
def index():
    generated_text = ""
    sentiment = ""
    
    if request.method == "POST":
        text = request.form["text"]
    

        # Text Generation
        generated_output = generator(text, max_length=100,max_new_tokens=100)
        generated_text = generated_output[0]['generated_text']

        # Sentiment Analysis
        sentiment_result = classifier(text)
        sentiment = sentiment_result[0]['label']
        
    return render_template("home.html", generated_text=generated_text, sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
