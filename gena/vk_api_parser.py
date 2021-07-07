import time
import requests
import pandas as pd


def get_jokes(new_domain: str, max_offset: int) -> list:
    """
    :param new_domain: строка с доменом паблика
    :param max_offset: максимальный номер поста
    :return:
    """
    serv = 'server_token'  # записать свой сервер токен из вк апи.
    version = '5.52'
    domain = new_domain
    count = 100  # парсим 100 постов за раз (больше нельзя)
    offset = 0  # отсуп от первого поста
    all_jokes = []
    while offset < max_offset:
        response = requests.get('https://api.vk.com/method/wall.get', params={
                            'access_token': serv,
                            'v': version,
                            'domain': domain,
                            'count': count,
                            'offset': offset
                        }
                        )
        data = response.json()['response']['items']
        offset += 100
        for post in data:
            all_jokes.append(post['text'])
        time.sleep(0.5)
    return all_jokes


"""
Пути к различным параметрам в json ответе. 
#marked_as_ads = data['response']['items'][0]['marked_as_ads']
#text = data['response']['items'][0]['text']
#comments_count = data['response']['items'][0]['comments']['count']
#likes_count = data['response']['items'][0]['likes']['count']
#reposts_count = data['response']['items'][0]['reposts']['count']
"""


""" 
Примеры доменов
(Обработано) 'baneksbest' - анекдоты категории Б: избранное # около 700 анекдотов ( всего в группе 715 записей)
'best__jokes'
"""


def file_writter(jokes: list, name: str):
    df = pd.DataFrame({'col': jokes})
    df.to_csv(name, encoding='utf-8')
