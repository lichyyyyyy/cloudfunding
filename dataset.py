import json
import pickle
import datetime
from functools import reduce


data = eval(open('final_updated', encoding='utf8').read())
pairs = []

for item in data.values():
	pair = []
	for (key, value) in item['series'].items():
		if (key-item['launch']).days < -1:
			print([(key-item['launch']).days, value])
		pair.append([(key-item['launch']).days, value])
	#print(pair)
	pairs.append(pair)

pickle.dump(pairs, open('dataset.p', 'wb'))
