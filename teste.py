import json


with open("companies.json") as file:
	companies = json.load(file)

labels = []

for intent in companies["intents"]:
    print(intent["tag"])
    print(intent["code"])

    labels.append(intent["tag"])
        

print(labels)