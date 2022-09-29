import requests
from bs4 import BeautifulSoup
import requests
result = requests.get('https://pycon.jp/2016/ja/schedule/talks/list/')
# result = requests.get('http://www.brain.kyutech.ac.jp/~furukawa/')
soup = BeautifulSoup(result.text, 'html.parser')
soup.find_all('img')



# result = requests.get('https://pycon.jp/2016/ja/schedule/talks/list/')
# print(result.text)

# with open('pycon_jp2017.txt', 'w') as f:
  # print(result.text, file=f)
print(soup.find('div', class_='presentation'))
# print(len(soup.find_all('div', class_='presentation')))


