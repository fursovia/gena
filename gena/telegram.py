import boto3
import random
import numpy as np
# import torch

# from bs4 import BeautifulSoup
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
from aiogram import Bot, types
from aiogram.utils import executor
from aiogram.dispatcher import FSMContext, Dispatcher
from aiogram.dispatcher.filters import Text, Filter
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage

bot = Bot(token='1868573125:AAGlocZJ6gu8ZASb4htCyf59vGBYj2iYYWs')
dp = Dispatcher(bot)

headers = {'X-API-KEY': '69fe63b7-6181-4972-8d37-8a2f0e459f7c'}
modes = ['Случайный анекдот', 'Задать начало']
ENDPOINT = "https://docapi.serverless.yandexcloud.net/ru-central1/b1g1fpqdg6m60mitk5ja/etn02r7q0t1hdfgh4m6r"
ACCESS_KEY = 'DZw7BIw5ooWxqz7FZjgj'#os.environ['ACCESS_KEY']
SECRET_KEY = 'CZQN9LcHQvuLpX5jC-gGtNUFVBnYK1ig4Tmd0QOT' #os.environ['SECRET_KEY']

np.random.seed(42)
# torch.manual_seed(42)


# def load_tokenizer_and_model(model_name_or_path):
#     return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()
#
#
# def generate(
#     model, tok, text,
#     do_sample=True, max_length=50, repetition_penalty=5.0,
#     top_k=5, top_p=0.95, temperature=1,
#     num_beams=None,
#     no_repeat_ngram_size=3
#     ):
#     input_ids = tok.encode(text, return_tensors="pt").cuda()
#     out = model.generate(
#       input_ids.cuda(),
#       max_length=max_length,
#       repetition_penalty=repetition_penalty,
#       do_sample=do_sample,
#       top_k=top_k, top_p=top_p, temperature=temperature,
#       num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
#       )
#     return list(map(tok.decode, out))


# tok, model = load_tokenizer_and_model("sberbank-ai/rugpt3small_based_on_gpt2")


n = 1

file = open('anecdotes.csv', encoding='utf-8')
read = file.readlines()
for i in range(len(read)):
    read[i] = read[i].replace('<br/>', '\n').replace('<br>', '\n').replace('</br>', '\n')


def load_user_info(user, table):
    ydb_docapi_client = boto3.resource('dynamodb', region_name='ru-central1', endpoint_url=ENDPOINT,
                                       aws_access_key_id=ACCESS_KEY,
                                       aws_secret_access_key=SECRET_KEY)
    table = ydb_docapi_client.Table(table)
    table.put_item(Item=user)


def get_user(user_id, table, hash_name):
    ydb_docapi_client = boto3.resource('dynamodb', region_name='ru-central1', endpoint_url=ENDPOINT,
                                       aws_access_key_id=ACCESS_KEY,
                                       aws_secret_access_key=SECRET_KEY)
    table = ydb_docapi_client.Table(table)
    response = table.get_item(Key = {hash_name: user_id})
    return response['Item']


def update_user(user_id, parameter, value, table, hash_name):
    ydb_docapi_client = boto3.resource('dynamodb', region_name='ru-central1', endpoint_url=ENDPOINT,
                                       aws_access_key_id=ACCESS_KEY,
                                       aws_secret_access_key=SECRET_KEY)
    table = ydb_docapi_client.Table(table)

    table.update_item(
        Key = {hash_name: user_id},
        UpdateExpression=f"set {parameter} = :p",
        ExpressionAttributeValues={
            ':p': value},
        ReturnValues="UPDATED_NEW"
    )


class RandomAnec(Filter):
    async def check(self, message: types.Message):
        parameters = get_user(message.chat.id, 'NeOleg', 'user_id')
        return parameters['mode'] == 0


class AnecByStart(Filter):
    async def check(self, message: types.Message):
        parameters = get_user(message.chat.id, 'NeOleg', 'user_id')
        return parameters['mode'] == 1


def CreateRankButton():
    markup = types.inline_keyboard.InlineKeyboardMarkup()
    markup.add(types.inline_keyboard.InlineKeyboardButton(text='Оррр выше гоооор!!! \U0001F44D',
                                                          callback_data='rate like'))
    markup.add(types.inline_keyboard.InlineKeyboardButton(text='Так себе \U0001F44E',
                                                          callback_data='rate dislike'))
    return markup


@dp.message_handler(commands=['start'])
async def send_welcome(message):
    await message.answer(f'*Привет, {message.from_user.first_name}!* '
                         f'*Я Не Олег*! Ты можешь поднять себе настроение, почитав мои уморительные анекдоты!',
                         parse_mode='Markdown')

    markup = types.reply_keyboard.ReplyKeyboardMarkup(one_time_keyboard=False, row_width=1, resize_keyboard=True)
    markup.add('Случайный анекдот', 'Задать начало')

    parameters = {'user_id': message.chat.id, 'mode': 0, 'anec': ''}
    load_user_info(parameters, 'NeOleg')
    await message.answer('Выбери способ генерации анекдотов.', reply_markup=markup)


@dp.message_handler(Text(modes), content_types=['text'])
async def process_step(message):
    global n, read
    if message.text == 'Случайный анекдот':
        markup = CreateRankButton()
        anec = read[random.choice(range(len(read)))]
        update_user(message.chat.id, 'anec', anec, 'NeOleg', 'user_id')
        update_user(message.chat.id, 'mode', 0, 'NeOleg', 'user_id')
        n += 1
        await message.answer(anec, reply_markup=markup)
    elif message.text == 'Задать начало':
        update_user(message.chat.id, 'mode', 1, 'NeOleg', 'user_id')
        await message.answer('Введите начало анекдота.')


@dp.message_handler(RandomAnec(), content_types=['text'])
async def get_film_by_year(message):
    await message.answer(f'Смените режим генерации анекдотов на *Задать начало*',
                         parse_mode='Markdown')


@dp.message_handler(AnecByStart(), content_types=['text'])
async def get_film_by_year(message):
    global n
    markup = CreateRankButton()

    await message.answer(f'{message.text}',
                         reply_markup=markup,
                         parse_mode='Markdown')

    update_user(message.chat.id, 'anec', message.text, 'NeOleg', 'user_id')
    n += 1

@dp.callback_query_handler(Text(startswith='rate'))
async def callback_chosen_film(call):
    global n
    rate = call.data.split()[1]
    anec = get_user(call.message.chat.id, 'NeOleg', 'user_id')['anec']
    parameters = {'code': n, 'user_id': call.message.chat.id, 'Anec': anec, 'Rate': rate}
    load_user_info(parameters, 'NeOlegLogging')
    await call.answer(f'Спасибо за оценку!!! \U0001F60D')




if __name__ == '__main__':
    executor.start_polling(dp)

