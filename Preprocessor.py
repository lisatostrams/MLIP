import re
import csv
from textblob import TextBlob
import json

def ReadFile(file):
	with open(file, encoding="utf8") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		var = []
		for row in csv_reader:
			var.append(row)
	return var

def OpenFile(file):
	with open(file) as json_file:
		data = json.load(json_file)
	return data
	
#No name = 0, name = 1
def preprocess(data):
	adjectives = ['energ','play','health','cute','love','sweet','beaut','friend','fun','ador','activ','good','great','best','abandon','adopt','vaccin','mix','black','white','vet','indoor','family','stray','injur','puppy','kitten','spayed']
	adjCount = [0] * len(adjectives)
	NoNameCount = 0
	NameCount = 0
	for i in range(len(data)):							# add empty columns to the matrix
		for k in range (len(adjCount)+3):
			data[i].append(k)
	for j in range(len(adjectives)):					# set the adjectives names in the first row
			data[0][24+j] = adjectives[j]
	for row in data:
		if (row[1].lower() == 'no name yet' or row[1].lower() == 'no name' or row[1].lower() == '' or row[1].lower() == 'not yet named' ):
			row[1] = 0
			NoNameCount += 1
		else:
			row[1] = 1
			NameCount += 1
		data[0][1] = 'Name'
		for i in range(len(adjectives)):				# set the position of the adjective to 1 if the describtion contains the adjective.
			if adjectives[i] in row[20].lower():
				row[24+i] = 1
				adjCount[i] += 1
		row[len(row)-1] = len(row[20])					# Description length
		blob = TextBlob(row[20])
		sentiment = blob.sentiment.polarity
		row[len(row)-3] = sentiment				# Sentiment score
	uniqueID = []								# from here popularity test resquer id
	for row in data:
		if row[18] not in uniqueID:
			uniqueID.append(row[18])
	for id in uniqueID:
		idcount = 0
		for row in data:
			if row[18] == id:
				idcount += 1
		for row in data:
			if row[18] == id:
				row[len(row)-2] = idcount
	data[0][52] = 'sentiment'
	data[0][53] = 'popularity resquer id'
	data[0][54] = 'description length'
	print(adjectives)
	print(adjCount)
	print("no name count: "+ str(NoNameCount))
	print("name count: " + str(NameCount))
	print("number of resquers : " + str(len(uniqueID)))
	return data
	
def write(data, filename):
	with open(filename, encoding = "utf8", mode='w',newline = '') as csv_file:
		writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_ALL)
		for row in data:
			writer.writerow(row)
	
filename = r"train.csv"
savefile = r"preprocessedTrain.csv"
data = ReadFile(filename)
newdata = preprocess(data)
write(newdata,savefile)