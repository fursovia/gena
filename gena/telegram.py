import re
import razdel
import random
import logging
import asyncio
import os
import functools
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from functools import lru_cache

import boto3
import torch
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from aiogram import Bot, types
from aiogram.utils import executor
from aiogram.dispatcher import FSMContext, Dispatcher
from aiogram.dispatcher.filters import Text, Filter


np.random.seed(42)
torch.manual_seed(42)

bot = Bot(token=os.environ['BOT_TOKEN'])
dp = Dispatcher(bot)

LOOP = asyncio.get_event_loop()

MAX_LENGTH = int(os.environ['MAX_LENGTH'])

ROOT_DIR = (Path(__file__).parent / "..").resolve()
ANECDOTES_FILE = ROOT_DIR / "data" / "anecdotica.csv"

MODES = ('Случайный анекдот', 'Задать начало')
MODES_GENERATION = ('RandAnec', 'RandStart', 'UserStart')
ENDPOINT = os.environ['ENDPOINT']
ACCESS_KEY = os.environ['ACCESS_KEY']
SECRET_KEY = os.environ['SECRET_KEY']
LOGS_FILE = os.environ.get("LOGS_FILE", "/home/bulat/gena/gena/file.log")
MODEL_DIR = os.environ.get("MODEL_DIR", "/home/bulat/gena/models/rugpt3small_based_on_gpt2_pretrained")

logger = logging.getLogger(__name__)
f_handler = logging.FileHandler(LOGS_FILE)
f_handler.setLevel(logging.DEBUG)
f_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(f_handler)


def check_for_digit(text: str) -> bool:
    try:
        temperature = float(text)
        return True
    except ValueError:
        return False


def process_final_anec(anec: str) -> str:
    residual = re.split('[.!?]', anec)[-1]
    return anec[:len(anec) - len(residual)]


def start_logging(
        user_id: str,
        anec: str,
        rate: int,
        modes_gen: int,
        logger: logging.Logger
) -> None:
    logger.warning(f'{user_id} <pp> {anec} <pp> {rate} <pp> {modes_gen}')


@lru_cache(maxsize=None)
def create_model() -> Tuple[GPT2Tokenizer, GPT2LMHeadModel]:
    tok, model = GPT2Tokenizer.from_pretrained(MODEL_DIR), GPT2LMHeadModel.from_pretrained(MODEL_DIR).cuda()
    return tok, model


def random_anec() -> List[str]:
    data = pd.read_csv(ANECDOTES_FILE)
    anec = random.choice(data.col)
    return anec


def generate(
        model: GPT2LMHeadModel,
        tok: GPT2Tokenizer,
        text: str,
        do_sample: bool = True,
        max_length: int = 50,
        repetition_penalty: float = 5.0,
        top_k: int = 5,
        top_p: float = 0.95,
        temperature: float = 1,
        num_beams: Optional[int] = None,
        no_repeat_ngram_size: int = 3
) -> List[str]:
    input_ids = tok.encode(text, return_tensors="pt").cuda()
    out = model.generate(
      input_ids.cuda(),
      max_length=max_length,
      repetition_penalty=repetition_penalty,
      do_sample=do_sample,
      top_k=top_k, top_p=top_p, temperature=temperature,
      num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
      )
    return list(map(tok.decode, out))


def load_user_info(
        user: Dict[str, Any],
        table: str
) -> None:
    ydb_docapi_client = boto3.resource('dynamodb', region_name='ru-central1', endpoint_url=ENDPOINT,
                                       aws_access_key_id=ACCESS_KEY,
                                       aws_secret_access_key=SECRET_KEY)
    table = ydb_docapi_client.Table(table)
    table.put_item(Item=user)


def get_user(
        user_id: str,
        table: str,
        hash_name: str
) -> Dict[str, Any]:
    ydb_docapi_client = boto3.resource('dynamodb', region_name='ru-central1', endpoint_url=ENDPOINT,
                                       aws_access_key_id=ACCESS_KEY,
                                       aws_secret_access_key=SECRET_KEY)
    table = ydb_docapi_client.Table(table)
    response = table.get_item(Key = {hash_name: user_id})
    return response['Item']


def update_user(
        user_id: str,
        parameter: str,
        value: Any,
        table: str,
        hash_name: str
) -> None:
    ydb_docapi_client = boto3.resource(
        'dynamodb',
        region_name='ru-central1',
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    table = ydb_docapi_client.Table(table)
    table.update_item(
        Key={hash_name: user_id},
        UpdateExpression=f"set {parameter} = :p",
        ExpressionAttributeValues={':p': value},
        ReturnValues="UPDATED_NEW"
    )


class RandomAnec(Filter):
    async def check(
            self,
            message: types.Message
    ) -> bool:
        parameters = get_user(message.chat.id, 'NeOleg', 'user_id')
        return parameters['mode'] == 0


class AnecByStart(Filter):
    async def check(
            self,
            message: types.Message
    ) -> bool:
        parameters = get_user(message.chat.id, 'NeOleg', 'user_id')
        return parameters['mode'] == 1


class Temperature(Filter):
    async def check(
            self,
            message: types.Message
    ) -> bool:
        return get_user(message.chat.id, 'NeOleg', 'user_id')['change_temperature'] == 1


def create_rank_button():
    markup = types.inline_keyboard.InlineKeyboardMarkup(one_time_keyboard=True)
    markup.add(types.inline_keyboard.InlineKeyboardButton(text='Оррр выше гоооор!!! \U0001F44D',
                                                          callback_data='rate like'))
    markup.add(types.inline_keyboard.InlineKeyboardButton(text='Так себе \U0001F44E',
                                                          callback_data='rate dislike'))
    return markup


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.answer(f'*Привет, {message.from_user.first_name}!* '
                         f'*Я Не Олег*! Ты можешь поднять себе настроение, почитав мои уморительные анекдоты!\n\n'
                         f'Если нужна помощь, то нажми /help.',
                         parse_mode='Markdown')

    markup = types.reply_keyboard.ReplyKeyboardMarkup(one_time_keyboard=False, row_width=1, resize_keyboard=True)
    markup.add(*MODES)

    parameters = {'user_id': message.chat.id, 'mode': 0, 'anec': '', 'modes_gen': 0, 'change_temperature': 0, 'temperature': '1.0'}
    load_user_info(parameters, 'NeOleg')
    await message.answer('Выбери способ генерации анекдотов.', reply_markup=markup)


@dp.message_handler(commands=['help'])
async def get_help(message):
    await message.answer(f'В нашем боте доступно 2 опции:\n\n'
                         f'<b>Случайный анекдот</b> - из базы будет выбрана случайная затравка (начало анекдота) и по ней '
                         f'сгенерирован анекдот.\n\n'
                         f'<b>Задать начало</b> - пользователь сам задает начало анекдота.'
                         f'\n\n\n'
                         f'Список доступных команд:\n'
                         f'/start - перезапустить бота (в случае неисправностей)\n'
                         f'/help - получить информацию о возможностях бота\n'
                         f'/change_temperature - изменить степень вариативности анекдотов',
                         parse_mode='HTML')


@dp.message_handler(commands=['change_temperature'])
async def chenge_temperature(message: types.Message):
    update_user(message.chat.id, 'change_temperature', 1, 'NeOleg', 'user_id')
    await message.answer(text='Выберите степень вариативности анекдотов (число больше 0). С увеличением степени вариативности '
                              'генерируемые анекдоты разнообразнее, но могут быть менее связными.')


@dp.message_handler(Temperature(), content_types=['text'])
async def get_top_n_films(message, state: FSMContext):
    if not check_for_digit(message.text):
        await message.answer(f'Введи число.')
    elif 0 >= float(message.text):
        await message.answer(f'Введи значение больше 0.')
    else:
        update_user(message.chat.id, 'temperature', message.text, 'NeOleg', 'user_id')
        update_user(message.chat.id, 'change_temperature', 0, 'NeOleg', 'user_id')

        await message.answer(f'Теперь степень вариативности равна {message.text}.')


@dp.message_handler(Text(MODES), content_types=['text'])
async def process_step(message: types.Message):
    if message.text == MODES[0]:
        markup = create_rank_button()
        anec = random_anec()
        mode_gen = np.random.choice(2, p=[0.3, 0.7])
        tok, model = create_model()
        temperature = float(get_user(message.chat.id, 'NeOleg', 'user_id')['temperature'])
        if mode_gen == 1:
            beginning = next(razdel.sentenize(anec)).text
            generated = await LOOP.run_in_executor(None, functools.partial(generate, model=model, tok=tok,
                                                                           text=beginning, num_beams=5, max_length=MAX_LENGTH,
                                                                           temperature=temperature))
            anec = process_final_anec(generated[0])
        update_user(message.chat.id, 'anec', anec, 'NeOleg', 'user_id')
        update_user(message.chat.id, 'mode', 0, 'NeOleg', 'user_id')
        update_user(message.chat.id, 'modes_gen', mode_gen, 'NeOleg', 'user_id')
        await message.answer(anec, reply_markup=markup)
    elif message.text == MODES[1]:
        update_user(message.chat.id, 'mode', 1, 'NeOleg', 'user_id')
        update_user(message.chat.id, 'modes_gen', 2, 'NeOleg', 'user_id')
        await message.answer('Введите начало анекдота.')


@dp.message_handler(RandomAnec(), content_types=['text'])
async def change_mode(message: types.Message):
    await message.answer(f'Смените режим генерации анекдотов на *Задать начало*',
                         parse_mode='Markdown')


@dp.message_handler(AnecByStart(), content_types=['text'])
async def get_anec_by_start(message: types.Message):
    markup = create_rank_button()
    tok, model = create_model()
    temperature = float(get_user(message.chat.id, 'NeOleg', 'user_id')['temperature'])
    generated = await LOOP.run_in_executor(None, functools.partial(generate, model=model, tok=tok,
                                                                   text=message.text, num_beams=5, max_length=MAX_LENGTH,
                                                                   temperature=temperature))
    anec = process_final_anec(generated[0])
    await message.answer(f'{anec}',
                         reply_markup=markup,
                         parse_mode='Markdown')

    update_user(message.chat.id, 'anec', anec, 'NeOleg', 'user_id')


@dp.callback_query_handler(Text(startswith='rate'))
async def callback_rate(call: types.CallbackQuery):
    rate = 0 if call.data.split()[1] == 'dislike' else 1
    user_data = get_user(call.message.chat.id, 'NeOleg', 'user_id')
    anec = user_data['anec']
    mode_gen = user_data['modes_gen']
    start_logging(call.message.chat.id, anec.replace('\n\r\n', '<br>').replace('\n', '<br>'), rate, mode_gen,logger)
    await call.answer(f'Спасибо за оценку!!! \U0001F60D')
    await call.message.edit_text(anec)


if __name__ == '__main__':
    executor.start_polling(dp)
