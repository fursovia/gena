import aiohttp
import asyncio
import json
import pandas as pd


all_jokes = []
async def get_jokes(new_domain: str, max_offset: int, count=100, offset=0) -> None:
    """
    :param new_domain: строка с доменом паблика
    :param max_offset: максимальный номер поста или максимальное отсупление от первого постаа
    :param count: количесво постов, которые получаем за один запрос. 100 - наибольшее значение.
    :param offset: первоначальный отсуп от первого поста
    :return: список строк-шуток
    """
    async with aiohttp.ClientSession() as session:
        serv = '9432aed69432aed69432aed639944ad106994329432aed6f4d6d6600acdbb75e0c25eb8' # 'server_token'  # записать свой сервер токен из вк апи.
        version = '5.52'  # версия текущая api
        domain = new_domain
        # count = 100  # парсим 100 постов за раз (больше нельзя)
        # offset = 0  # отсуп от первого поста
        params = {'access_token': serv,
                  'v': version,
                  'domain': domain,
                  'count': count,
                  'offset': offset
                  }

        while offset < max_offset:
            async with session.get('https://api.vk.com/method/wall.get', params=params) as response:
                # data = response.json()['response']['items']
                data = await response.text()
                data_json = json.loads(data)
                data_json = data_json['response']['items']
                # data = r.text
                offset += count
                for post in data_json:
                    all_jokes.append(post['text'])
                await asyncio.sleep(0.5)
    # return all_jokes

loop = asyncio.get_event_loop()
loop.run_until_complete(get_jokes('best__jokes', 10, count=5))
print(all_jokes, len(all_jokes))
# print(all_jokes['response']['items'])

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


file_writter(all_jokes, 'async_try.csv')
