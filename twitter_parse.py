import requests
import json
import urllib2
from textblob import TextBlob
import csv
import pandas


#access_token = 'AAAAAAAAAAAAAAAAAAAAAPIl%2BQAAAAAAL4shfRRUwIqN2OMBTssw%2BwmxGh8%3DJGtRZKa7kndT2DyNO8MezacEM9dj9TYUcF4zsDQzm9G7b8lg7K'
access_token = 'AAAAAAAAAAAAAAAAAAAAAGxA%2BQAAAAAAzY%2FWIQ%2BNU4r0zEwTuYw%2BHW%2BnXMo%3DPKtNCi3C4FxY3q7trrgZEdfRTELgFpE8yFtPvKCHTJvcfdVrBS'
# access_token = 'AAAAAAAAAAAAAAAAAAAAAJkn%2BQAAAAAAx1a8%2FN%2BB5Y1IUQYdrt3OE5lwpgs%3DYDrGMzdq3yJYR91Hr72FwvXa2HW2f7qqnaOSH824pfB7F7Wt4D'
url = 'https://api.twitter.com/1.1/tweets/search/fullarchive/NBA.json'


def extract_emojis(str):
  return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)


colnames = ['name', 'season']
data = pandas.read_csv('nba_data.csv', names=colnames)
names = data.name.tolist()
seasons = data.season.tolist()

# print(names[index])
# print(seasons[index])

for index in range(122, 134):
	

	data = '{"query":"' + names[index] + ' lang:en", "maxResults": "100", "fromDate":"200808010000","toDate":"200907010000"}'
	req = urllib2.Request(url, data, {'Content-Type': 'application/json', 'Authorization': 'Bearer {}'.format(access_token)})
	response = urllib2.urlopen(req)
	data = json.load(response)
	with open('tweets_final2009.csv', 'a') as csvFile:
		writer = csv.writer(csvFile)
		for song in data['results']:   
			elem = song['text']
			elem = elem.replace("RT", "")
			# elem = extract_emojis(elem)
			elem2 = elem.split()
			if elem2 != '':
				result = []
				for item in elem2:
					if item[0] != "@":
						if "https://t" not in item: 
							result.append(item)

				seperator = ' '
				final = seperator.join(result)
				final = ''.join([i if ord(i) < 128 else ' ' for i in final])

				analysis = TextBlob(final)
				row = [final, analysis.sentiment.polarity, analysis.sentiment.subjectivity]
				print(final)
				print(analysis.sentiment)
				writer.writerow(row)
		writer.writerow([names[index], '_____________', '_______________'])
	csvFile.close()


