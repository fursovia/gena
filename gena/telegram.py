import re
import random
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from functools import lru_cache

import boto3
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from aiogram import Bot, types
from aiogram.utils import executor
from aiogram.dispatcher import FSMContext, Dispatcher
from aiogram.dispatcher.filters import Text, Filter

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

bot = Bot(token=os.environ['BOT_TOKEN'])
dp = Dispatcher(bot)

MAX_LENGTH = 50

ROOT_DIR = (Path(__file__).parent / "..").resolve()
ANECDOTES_FILE = ROOT_DIR / "data" / "anecdotes.csv"

MODES = ('Случайный анекдот', 'Задать начало')
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


def process_final_anec(anec):
    residual = re.split('[.!?]', anec)[-1]
    return anec[:len(anec) - len(residual)]


def start_logging(
        user_id: str,
        anec: str,
        rate: int,
        logger: logging.Logger
) -> None:
    logger.warning(f'{user_id} <pp> {anec} <pp> {rate}')


@lru_cache(maxsize=None)
def create_model() -> Tuple[GPT2Tokenizer, GPT2LMHeadModel]:
    tok, model = GPT2Tokenizer.from_pretrained(MODEL_DIR), GPT2LMHeadModel.from_pretrained(MODEL_DIR).cuda()
    return tok, model


def create_dataset_of_rand_anec() -> List[str]:
    data = []
    with ANECDOTES_FILE.open(encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().replace('<br/>', '\n').replace('<br>', '\n').replace('</br>', '\n')
            data.append(line)

    return data


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
                         f'*Я Не Олег*! Ты можешь поднять себе настроение, почитав мои уморительные анекдоты!',
                         parse_mode='Markdown')

    markup = types.reply_keyboard.ReplyKeyboardMarkup(one_time_keyboard=False, row_width=1, resize_keyboard=True)
    markup.add(*MODES)

    parameters = {'user_id': message.chat.id, 'mode': 0, 'anec': ''}
    load_user_info(parameters, 'NeOleg')
    await message.answer('Выбери способ генерации анекдотов.', reply_markup=markup)


@dp.message_handler(Text(MODES), content_types=['text'])
async def process_step(message: types.Message):
    anecdotes = create_dataset_of_rand_anec()
    if message.text == MODES[0]:
        markup = create_rank_button()
        anec = random.choice(anecdotes)
        update_user(message.chat.id, 'anec', anec, 'NeOleg', 'user_id')
        update_user(message.chat.id, 'mode', 0, 'NeOleg', 'user_id')
        await message.answer(anec, reply_markup=markup)
    elif message.text == MODES[1]:
        update_user(message.chat.id, 'mode', 1, 'NeOleg', 'user_id')
        await message.answer('Введите начало анекдота.')


@dp.message_handler(RandomAnec(), content_types=['text'])
async def change_mode(message: types.Message):
    await message.answer(f'Смените режим генерации анекдотов на *Задать начало*',
                         parse_mode='Markdown')


@dp.message_handler(AnecByStart(), content_types=['text'])
async def get_anec_by_start(message: types.Message):
    markup = create_rank_button()
    tok, model = create_model()
    generated = generate(model, tok, message.text, num_beams=5, max_length=MAX_LENGTH)
    anec = process_final_anec(generated[0])
    await message.answer(f'{anec}',
                         reply_markup=markup,
                         parse_mode='Markdown')

    update_user(message.chat.id, 'anec', anec, 'NeOleg', 'user_id')


@dp.callback_query_handler(Text(startswith='rate'))
async def callback_rate(call: types.CallbackQuery):
    rate = 0 if call.data.split()[1] == 'dislike' else 1
    anec = get_user(call.message.chat.id, 'NeOleg', 'user_id')['anec']
    start_logging(call.message.chat.id, anec.replace('\n', '<br>'), rate, logger)
    await call.answer(f'Спасибо за оценку!!! \U0001F60D')
    await call.message.edit_text(anec)


if __name__ == '__main__':
    executor.start_polling(dp)
