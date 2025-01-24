""" Scraping information from Wikipedia and saving it to a CSV file. """

import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = 'https://en.wikipedia.org/wiki/Cloud-computing_comparison'
response = requests.get(URL, timeout=30)

# if response.status_code == 200:
#     print('Request is successful.')
# else:
#     print('Request returned an error.')

soup = BeautifulSoup(response.content, 'html.parser')

print("Title: " + soup.title.text)

table = soup.find('table')

rows = table.find_all('tr')

headers = [header.text.strip() for header in rows[0].find_all('th')]

data = []

for row in rows[1:]:
    cols = row.find_all('td')
    cols = [col.text.strip() for col in cols]
    data.append(cols)

df = pd.DataFrame(data, columns=headers)

print(df.head())

df.to_csv('cloud_computing_comparison.csv', index=False)
