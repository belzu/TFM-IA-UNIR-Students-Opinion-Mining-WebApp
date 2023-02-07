from pandas.io.spss import Path
import pandas as pd
import numpy as np
import sklearn, re, glob
import unicodedata
import nltk
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
import collections
import stanza, copy
from nltk import word_tokenize          
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import metrics
#from joblib import dump, load
import joblib
#import pickle
import json
from datetime import datetime

class DataManager:
	
	synonyms = {
				"docente": r"(profesora?|educadora?|maestr(o|a)|pedagog(o|a)|instructora?|preceptora?)",
				"malo": r"(mala|horrible|lamentable|decepci(onante|ón)|desorganizad(o|a)|incult(o|a)|penos(o|a)|p(é|e)sim(o|a)|deficiente|defectuos(o|a)|imperfect(o|a)|nefast(o|a)|desastr(e|os(o|a))|perjudicial|cruel|despiadad(o|a)|paup(é|e)rrim(o|a)|in(ú|u)til|inservible|incapaz|incompetente|inh(á|a)bil|inept(o|a))",				
				"bueno": r"(buena|maravillos(o|a)|organizad(o|a)|eficiente|virtuos(o|a)|genial|fant(á|a)stic(o|a)|correct(o|a)|magn(á|a)nim(o|a)|adecuad(o|a)|(í|i)ntegr(o|a)|(poco|nada|cero) mal(o|a)?)",
				"muy bueno": r"(poco|(para )?nada|en absoluto) malo",
				"muy malo": r"(poco|(para )?nada|en absoluto) bueno"
				}

	#Code obtained from https://micro.recursospython.com/recursos/como-quitar-tildes-de-una-cadena.html
	def remove_accents_in_word(word):
		replaced_vocals = (
			("á", "a"),
			("é", "e"),
			("í", "i"),
			("ó", "o"),
			("ú", "u"),
		)
		for accented, not_accented in replaced_vocals:
			word = word.replace(accented, not_accented).replace(accented.upper(), not_accented.upper())
		return word

	def read_comments_without_sentiments(path):
		print("Leyendo comentarios de " + path)
		df = pd.read_csv(path, delimiter='\t')
		return df["Comentario"].values.tolist()

	def read_comments_with_sentiments(path):
		print("Leyendo comentarios de " + path)
		df = pd.read_csv(path, delimiter='\t')
		comments = df["Comentario"].values.tolist()
		sentiments = df["Sentimiento"].values.tolist()
		return [[comments[i], int(sentiments[i])] for i in range(len(comments))]
		 

	def obtain_preprocessed_comments_with_sentiments(comments, correct=True, lemmatize=True, synonymize = False, snlp = None):
		#Se obtienen los comentarios preprocesados
		preprocessed_comments = DataManager.preprocess_comments([comment[0] for comment in comments], correct, lemmatize, synonymize, snlp)
		#Se obtienen los sentimientos de los comentarios
		sentiments = [comment[1] for comment in comments]
		#Se devuelven ambos
		return preprocessed_comments, sentiments

	def obtain_preprocessed_comments_without_sentiments(comments, correct=True, lemmatize=True, synonymize = False, snlp = None):
		#Se obtienen los comentarios preprocesados
		return DataManager.preprocess_comments(comments, correct, lemmatize, synonymize, snlp)

	'''
	Se realiza un pre-procesado de los comentarios que involucra corregir y lematizar palabras así como pasar todas las palabras a minuscula
	:param correct: True si se desea correccion de palabras, false si no
	:param lemmatize: True si se desea trabajar con los lemas de las palabras, false si no
	'''
	def preprocess_comments(comments, correct=True, lemmatize=True, synonymize = False, snlp = None):
		#Se pasan todos los comentarios a minuscula
		comments = [comment.lower() for comment in comments]
		#Se corrigen las palabras mal escritas, aunque hay que tener cuidado, pues puede proporcionar errores
		if correct:
			comments = DataManager.correct_mispelt_words(comments)
		#Se aplica la sinonimización
		if synonymize:
			DataManager.synonymize(comments)
		#Se aplica la lematizacion
		if lemmatize:
			comments = DataManager.lemmatize_words(comments, snlp)
		return comments

	def synonymize(comments):
		for idx,comment in enumerate(comments):
			for synonym in DataManager.synonyms:
				#print("Antes " + comments[idx])
				comments[idx] = re.sub(DataManager.synonyms[synonym], synonym, comments[idx])
				#print("Despues " + comments[idx])

	#Corrige todas las palabras que esten mal escritas (en este caso token es cada palabra o simbolo que compone un comentario)
	def correct_mispelt_words(comments, lang = 'es'):
		spell = SpellChecker(language=lang)
		for idx,comment in enumerate(comments):
			tokens = DataManager.get_all_tokens_in_comment(comment)
			misspelt = spell.unknown(tokens)
			if len(misspelt)>0:
				for token in misspelt:
					if token is not None and spell.correction(token) is not None:
						comments[idx] = re.sub(r"\b{}\b".format(re.escape(token)), spell.correction(token), comments[idx])
		return comments

	def get_all_tokens_in_comment(comment):
		tokenizer = RegexpTokenizer(r'\w+')
		return tokenizer.tokenize(comment)

	def lemmatize_words(comments, snlp):
		#Coge el vocabulario y sustituye cada palabra por su palabra lematizada
		for cidx, comment  in enumerate(comments):
			comment_tokens = snlp(comment)
			word_tokens = [token.text for sent in comment_tokens.sentences for token in sent.tokens]
			word_lemmas = [word.lemma for sent in comment_tokens.sentences for word in sent.words]
			text_and_lemmas = [(word.text, word.lemma) for sent in comment_tokens.sentences for word in sent.words]
			# TODO Se lematizan palabras en el conjunto de tokens que a StanzaLanguage se le hayan pasado por alto
			#Se lematizan todas las palabras de cada comentario (o documento)
			for text, lemma in text_and_lemmas:
				comments[cidx] = re.sub(r"\b{}\b".format(re.escape(str(text))), str(lemma).lower(), comments[cidx])
		return comments

	def save_comments(comments, sentiments, path = "./", filename = None):
		filename = DataManager.obtain_filename(filename)
		with open(path+filename+".dat", "wb") as outfile:
			# "wb" argument opens the file in binary mode
			pickle.dump((comments, sentiments), outfile)

	def load_comments(filename):
		with open(filename, "rb") as file:
			# "wb" argument opens the file in binary mode
			return pickle.load(file)	

	def obtain_filename(filename):
		if filename == None:
			return DataManager.get_current_date_time_str()
		else:
			return filename

	def get_current_date_time_str():
		now = datetime.now()
		dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
		dt_string = dt_string.replace("/","-")
		dt_string = dt_string.replace(" ","_")
		dt_string = dt_string.replace(":","_")
		return str(dt_string)


class SentimentManager:

	def __init__(self, data_representation_model = CountVectorizer(), data_classification_model = svm.SVC(), snlp = stanza.Pipeline(lang="es")):
		self.data_representation_model = data_representation_model
		#stop_words = [DataManager.remove_accents_in_word(i) for i in stopwords.words('spanish')]
		#stop_words.extend(stopwords.words('spanish'))
		#self.data_representation_model.stopwords = stop_words
		self.data_classification_model = data_classification_model
		self.snlp = snlp
	
	def train_with_comments_csv(self, ruta_csvs):
		self.train_with_comments(DataManager.read_comments_with_sentiments(ruta_csvs))

	def train_with_comments(self, comments, sentiments):
		count_array = self.transform_comments_into_count_array(comments)
		self.train(count_array, sentiments)
  
	#Convierte los comentarios a formato numerico
	def transform_comments_into_count_array(self, comments):
		count_matrix = self.data_representation_model.fit_transform(comments)
		count_array = count_matrix.toarray()
		return count_array

	#Entrena el modelo a partir del count_array de los comentarios (X_train) y de los sentimientos (y_train)
	def train(self, X_train, y_train):
		self.data_classification_model.fit(X_train, y_train)


	def obtain_predictions(self, comments, correct=True, lemmatize=True, synonymize = True):
		sentiments = self.predict_comments(comments, correct, lemmatize, synonymize).tolist()
		result = [{"comment":i[0], "sentiment":i[1]} for i in zip(comments, sentiments)]
		stats = [sentiments.count(0), sentiments.count(1), sentiments.count(2)]
		result.append({"number_of_negatives": stats[0], "number_of_neutrals": stats[1],
		               "number_of_positives": stats[2] , "most_frequent": stats.index(max(stats))})
		return result

	def predict_comments(self, comments, correct=True, lemmatize=True, synonymize = True):
		preprocessed_comments = DataManager.obtain_preprocessed_comments_without_sentiments(comments, correct, lemmatize, synonymize, self.snlp)
		return self.predict(self.data_representation_model.transform(preprocessed_comments).toarray())
  
	def predict(self, X):
		return self.data_classification_model.predict(X)

	def get_accuracy(self, comments, correct = True, lemmatize = True, synonymize = False, test_size = 0.3, random_state = None):
		preprocessed_comments, sentiments = DataManager.obtain__preprocessed_comments_with_sentiments(comments, correct, lemmatize, synonymize, self.snlp)
		#Se obtiene el modelo de representacion de datos (el modelo que representa los comentarios de forma numerica)
		count_matrix = self.data_representation_model.fit_transform(preprocessed_comments)
		count_array = count_matrix.toarray()
		# El 30% de los datos será para testear, el 70% para entrenar
		X_train, X_test, y_train, y_test = train_test_split(count_array, sentiments, test_size=test_size, random_state=random_state) # 70% training and 30% test
		self.data_classification_model.fit(X_train, y_train)
		y_pred = self.data_classification_model.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		return y_pred, acc

	def load_model(filename):
		model = SentimentManager()
		model.data_representation_model, model.data_classification_model = joblib.load(filename)
		return model
	
	def save_model(self, path = "./", filename = None):
		if filename == None:
			filename = DataManager.get_current_date_time_str()
		with open(path+filename + ".joblib", 'wb') as fout:
			joblib.dump((self.data_representation_model, self.data_classification_model), fout)
