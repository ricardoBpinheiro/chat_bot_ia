# pip install nltk
# pip install tensorflow
# pip install numpy
# pip install tflearn

# conda activate chatbot_ia

import pickle
import json
import requests
import random
import tensorflow
import tflearn
import numpy
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from nltk import tokenize 
import os

KEY = 'd58315ae'
URL = 'https://api.hgbrasil.com/finance/'

with open("intents.json") as file:
	data = json.load(file)


with open("companies.json") as file:
	companies = json.load(file)


try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
        	# wrds = nltk.word_tokenize(pattern, language='portuguese', preserve_line=False)
            wrds = tokenize.word_tokenize(pattern, language='portuguese')
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

os.system('cls' if os.name == 'nt' else 'clear')

# Verifica se tem um modelo existente para nao repetir o treinamento
print("Caso o bot nunca tenha sido treinado escreva sim, caso o contrario escreva nao")
train_model_inp = input("Deseja Treinar o modelo ? (s/n): ")

if train_model_inp.lower() == "s" or train_model_inp.lower() == "sim":
    model.fit(training, output, n_epoch=10000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
else:
    try:
        model.load("model.tflearn")
    except:
        model.fit(training, output, n_epoch=10000, batch_size=8, show_metric=True)
        model.save("model.tflearn")

os.system('cls' if os.name == 'nt' else 'clear')


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s, language='portuguese')
    # s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def verify_company(word):
    for intent in companies["intents"]:
        if intent["tag"] == word:
            return intent["code"]

    return ""
            

def info_bolsa(code):
    url = f'{URL}stock_price/?key={KEY}&symbol={code}'
    # url = 'https://api.hgbrasil.com/finance?key=d58315ae'
    # print(url)
    response = requests.get(url=url)

    if 200 <= response.status_code <= 299:
        # Success
        return response.json()
    else:
        # Errors
        return ""


def format_data(data, company_symbol):
    aux =  data[company_symbol.upper()]
    data_formatted = f"Código: {aux['symbol']} \n" \
                     f"Nome: {aux['name']} \n" \
                     f"Preço: {aux['price']} \n" \
                     f"Variação: {aux['change_percent']}% \n" \
                     f"Capital de Mercado: {aux['market_cap']} \n" \
                     f"Ultima atualização: {aux['updated_at']}"
    return data_formatted


def help_options():
    data = "list -a             listar todas empresas \n" \
           "list -c company     listar empresa especifica (uso: list -c wege3||weg) \n"

    return data


def chat():
    print("Comece a escrever! (Para sair digite 'sair')")
    while True:
        inp = input("Você: ")
        if inp.lower() == "sair":
            break

        if inp.lower() != "":
            # Faz a comparação de probabilidade de todos os valores na lista (Cada neuronio do modelo)
            results = model.predict([bag_of_words(inp, words)])
            # print(results)

            results_index = numpy.argmax(results)
            tag = labels[results_index]
            # print(tag)

            company_code = verify_company(tag)
            # print(company_code)

            responses = ""
            company_info = ""
            company_data = ""
            if company_code != "":
                company_info = info_bolsa(company_code)

            if company_info != "":
                company_data = format_data(company_info["results"], company_code)

            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
            
            if responses != "":
                print(f"RonaldoTech: {random.choice(responses)}")

                if tag == "opcoes":
                    print(help_options())

                if company_data != "":
                    print("-"*40)
                    print(company_data)
                    print("-"*40)
            else:
                print(f"RonaldoTech: O que você disse?")
            
        else:
            print("RonaldoTech: Tem alguem ai ?")
            
chat()