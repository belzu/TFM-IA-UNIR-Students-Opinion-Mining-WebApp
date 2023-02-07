from flask import Flask, request, jsonify
from flask_cors import CORS
from Sentiment_evaluator import SentimentManager
from Sentiment_evaluator import DataManager
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)

PORT = 8000
DEBUG = True
ALLOWED_EXTENSIONS = set(['csv'])

model = SentimentManager.load_model("./models/BEST_F1_PASSIVE_AGGRESIVE_GRID_SEARCH_CV_TFIDF_V2.joblib")

def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(404)
def not_found(error):
	return "Not found!"


@app.route('/predict', methods = ['POST'])
def process():
	file = request.files['file']
	print(file.filename)	
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		filepath = os.path.join('input', filename)
		file.save(filepath)
		predictions = predict(filepath)
		os.remove(filepath)
		return jsonify(predictions)


def predict(filepath):
	comments = DataManager.read_comments_without_sentiments(filepath)
	return model.obtain_predictions(comments)


if __name__ == '__main__':
	app.run(port = PORT, debug = DEBUG)