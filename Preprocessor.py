import re
import csv

def ReadFile(file):
	with open(file, encoding="utf8") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		var = []
		for row in csv_reader:
			var.append(row)
	return var
	
#No name = 0, name = 1
def preprocess(data):
	adjectives = ['energ','play','health','cute','love','sweet','beaut','friend','fun','ador','activ','good','great','best','abandon','adopt','vaccin']
	adjCount = [0] * len(adjectives)
	NoNameCount = 0
	NameCount = 0
	for i in range(len(data)):							# add empty columns to the matrix
		for count in adjCount:
			data[i].append(count)
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
			if adjectives[i] in row[20]:
				row[24+i] = 1
				adjCount[i] += 
	print(adjectives)
	print(adjCount)
	print("no name count: "+ str(NoNameCount))
	print("name count: " + str(NameCount))
	return data
	
def write(data, filename):
	with open(filename, encoding = "utf8", mode='w',newline = '') as csv_file:
		writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_ALL)
		for row in data:
			writer.writerow(row)
	
filename = r"C:\Users\sjors\MLIP\train.csv"
savefile = r"C:\Users\sjors\MLIP\preprocessedTrain.txt"
data = ReadFile(filename)
newdata = preprocess(data)
write(newdata,savefile)