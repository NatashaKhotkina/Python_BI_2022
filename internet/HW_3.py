#!/usr/bin/env python
# coding: utf-8

# # Задание 1 (6 баллов)

# В данном задании мы будем работать со [списком 250 лучших фильмов IMDb](https://www.imdb.com/chart/top/?ref_=nv_mp_mv250)
# 
# 1. Выведите топ-4 *фильма* **по количеству оценок пользователей** и **количество этих оценок** (1 балл)
# 2. Выведите топ-4 лучших *года* (**по среднему рейтингу фильмов в этом году**) и **средний рейтинг** (1 балл)
# 3. Постройте отсортированный **barplot**, где показано **количество фильмов** из списка **для каждого режисёра** (только для режиссёров с более чем 2 фильмами в списке) (1 балл)
# 4. Выведите топ-4 самых популярных *режиссёра* (**по общему числу людей оценивших их фильмы**) (2 балла)
# 5. Сохраните данные по всем 250 фильмам в виде таблицы с колонками (name, rank, year, rating, n_reviews, director) в любом формате (2 балла)
# 
# Использовать можно что-угодно, но полученные данные должны быть +- актуальными на момент сдачи задания

# In[1]:


import requests
import re
import pandas as pd
from bs4 import BeautifulSoup


# ### Part 5

# In[2]:


response = requests.get("https://www.imdb.com/chart/top/?ref_=nv_mp_mv250")


# In[3]:


soup = BeautifulSoup(response.content, "lxml")


# In[4]:


title_columns = soup.find_all("td", class_="titleColumn")


# In[5]:


name = pd.Series([title_column.find("a").text for title_column in title_columns])


# In[7]:


rank = pd.Series(range(1, 251))


# In[8]:


year = pd.Series([int(element.text.strip("()")) 
                  for element in soup.find_all("span", class_="secondaryInfo")])


# In[9]:


rating = pd.Series([float(element.text) for element in soup.find_all("strong")])


# In[10]:


user_srings = [element.attrs["title"] for element in soup.find_all("strong")]


# In[11]:


n_reviews = pd.Series([int(user_sring.split(" ")[3].replace(",", "")) 
                       for user_sring in user_srings])


# In[15]:


director = pd.Series([title_column.find("a").attrs["title"].split(' (dir.)')[0] for title_column in title_columns])


# In[32]:


top_movies = {'Name': name, 'Rank': rank, 'Year': year, 'Rating': rating, 
              'N_reviews': n_reviews, 'Director': director}


# In[33]:


top_movies = pd.DataFrame(top_movies)


# In[34]:


top_movies.head()


# ### Part 1

# In[26]:


top_movies.sort_values(by='N_reviews', ascending=False).iloc[:4, [0, 4]]


# ### Part 2

# In[31]:


top_movies.groupby('Year', as_index=False)['Rating'] \
          .mean() \
          .sort_values(by='Rating', ascending=False) \
          .head(4)


# ### Part 3

# In[46]:


top_movies.groupby('Director', as_index=False)['Name'] \
          .count() \
          .rename(columns={'Name': 'N_films'}) \
          .query('N_films > 2') \
          .sort_values(by='N_films', ascending=False) \
          .plot.bar(x='Director', y='N_films');


# ### Part 4

# In[50]:


top_movies.groupby('Director', as_index=False)['N_reviews'] \
          .sum() \
          .sort_values(by='N_reviews', ascending=False) \
          .head(4)


# # Задание 2 (10 баллов)

# Напишите декоратор `telegram_logger`, который будет логировать запуски декорируемых функций и отправлять сообщения в телеграм.
# 
# 
# Вся информация про API телеграм ботов есть в официальной документации, начать изучение можно с [этой страницы](https://core.telegram.org/bots#how-do-bots-work) (разделы "How Do Bots Work?" и "How Do I Create a Bot?"), далее идите в [API reference](https://core.telegram.org/bots/api)
# 
# **Основной функционал:**
# 1. Декоратор должен принимать **один обязательный аргумент** &mdash; ваш **CHAT_ID** в телеграме. Как узнать свой **CHAT_ID** можно найти в интернете
# 2. В сообщении об успешно завершённой функции должны быть указаны её **имя** и **время выполнения**
# 3. В сообщении о функции, завершившейся с исключением, должно быть указано **имя функции**, **тип** и **текст ошибки**
# 4. Ключевые элементы сообщения должны быть выделены **как код** (см. скриншот), форматирование остальных элементов по вашему желанию
# 5. Время выполнения менее 1 дня отображается как `HH:MM:SS.μμμμμμ`, время выполнения более 1 дня как `DDD days, HH:MM:SS`. Писать форматирование самим не нужно, всё уже где-то сделано за вас
# 
# **Дополнительный функционал:**
# 1. К сообщению также должен быть прикреплён **файл**, содержащий всё, что декорируемая функция записывала в `stdout` и `stderr` во время выполнения. Имя файла это имя декорируемой функции с расширением `.log` (**+3 дополнительных балла**)
# 2. Реализовать предыдущий пункт, не создавая файлов на диске (**+2 дополнительных балла**)
# 3. Если функция ничего не печатает в `stdout` и `stderr` &mdash; отправлять файл не нужно
# 
# **Важные примечания:**
# 1. Ни в коем случае не храните свой API токен в коде и не загружайте его ни в каком виде свой в репозиторий. Сохраните его в **переменной окружения** `TG_API_TOKEN`, тогда его можно будет получить из кода при помощи `os.getenv("TG_API_TOKEN")`. Ручное создание переменных окружения может быть не очень удобным, поэтому можете воспользоваться функцией `load_dotenv` из модуля [dotenv](https://pypi.org/project/python-dotenv/). В доке всё написано, но если коротко, то нужно создать файл `.env` в текущей папке и записать туда `TG_API_TOKEN=<your_token>`, тогда вызов `load_dotenv()` создаст переменные окружения из всех переменных в файле. Это довольно часто используемый способ хранения ключей и прочих приватных данных
# 2. Функцию `long_lasting_function` из примера по понятным причинам запускать не нужно. Достаточно просто убедится, что большие временные интервалы правильно форматируются при отправке сообщения (как в примерах)
# 3. Допустима реализация логирования, когда логгер полностью перехватывает запись в `stdout` и `stderr` (то есть при выполнении функций печать происходит **только** в файл)
# 4. В реальной жизни вам не нужно использовать Telegram API при помощи ручных запросов, вместо этого стоит всегда использовать специальные библиотеки Python, реализующие Telegram API, они более высокоуровневые и удобные. В данном задании мы просто учимся работать с API при помощи написания велосипеда.
# 5. Обязательно прочтите часть конспекта лекции про API перед выполнением задания, так как мы довольно поверхностно затронули это на лекции
# 
# **Рекомендуемые к использованию модули:**
# 1. os
# 2. sys
# 3. io
# 4. datetime
# 5. requests
# 6. dotenv
# 
# **Запрещённые модули**:
# 1. Любые библиотеки, реализующие Telegram API в Python (*python-telegram-bot, Telethon, pyrogram, aiogram, telebot* и так далле...)
# 2. Библиотеки, занимающиеся "перехватыванием" данных из `stdout` и `stderr` (*pytest-capturelog, contextlib, logging*  и так далле...)
# 
# 
# 
# Результат запуска кода ниже должен быть примерно такой:
# 
# ![image.png](attachment:620850d6-6407-4e00-8e43-5f563803d7a5.png)
# 
# ![image.png](attachment:65271777-1100-44a5-bdd2-bcd19a6f50a5.png)
# 
# ![image.png](attachment:e423686d-5666-4d81-8890-41c3e7b53e43.png)

# In[2]:


import requests
import sys
import io
from datetime import datetime
from dotenv import dotenv_values
from io import StringIO 


# In[3]:


token = dotenv_values()['TG_API_TOKEN']


# In[5]:


chat_id = 265983504


# In[19]:


def telegram_logger(chat_id):
    def decorator(func):
        def inner_func(*args, **kwargs):
            file_stdout_err = StringIO()
            sys.stdout = file_stdout_err
            sys.stderr = file_stdout_err
            try:
                start = datetime.now()
                func(*args, **kwargs)
                stop = datetime.now()
                execution_time = str(stop - start)
                success = True
            except Exception as inst:
                success = False
                exc = inst
                exc_type = type(inst).__name__
            func_name = func.__name__
            if success:
                text = f"💃 Function `{func_name}` successfully finished in `{execution_time}`"
            else:
                text = f"💔 Function `{func_name}` failed with an exception: \n `{exc_type}`: `{exc}`"
            
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            # check wether something was written to file
            if len(file_stdout_err.getvalue()) > 0:
                file_stdout_err.seek(0)
                requests.post(f'https://api.telegram.org/bot{token}/sendDocument', 
                            params={'chat_id': chat_id, 'caption': text, 'parse_mode': 'MarkdownV2'},
                            files={'document': (f'{func_name}.log', file_stdout_err)})
            else:
                requests.get(f'https://api.telegram.org/bot{token}/sendMessage', 
                          params={'chat_id': chat_id, 'text': text, 'parse_mode': 'MarkdownV2'})
        return inner_func
    return decorator


# In[20]:


import time


@telegram_logger(chat_id)
def good_function():
    print("This goes to stdout")
    print("And this goes to stderr", file=sys.stderr)
    time.sleep(2)
    print("Wake up, Neo")

@telegram_logger(chat_id)
def bad_function():
    print("Some text to stdout")
    time.sleep(2)
    print("Some text to stderr", file=sys.stderr)
    raise RuntimeError("Ooops, exception here!")
    print("This text follows exception and should not appear in logs")
    
@telegram_logger(chat_id)
def long_lasting_function():
    time.sleep(200000000)


@telegram_logger(chat_id)
def no_print():
    time.sleep(5)
    
good_function()

try:
    bad_function()
except Exception:
    pass

no_print()

# long_lasting_function()


# # Задание 3
# 
# В данном задании от вас потребуется сделать Python API для какого-либо сервиса
# 
# В задании предложено два варианта: простой и сложный, **выберите только один** из них.
# 
# Можно использовать только **модули стандартной библиотеки** и **requests**. Любые другие модули можно по согласованию с преподавателем.

# ❗❗❗ В **данном задании** требуется оформить код в виде отдельного модуля (как будто вы пишете свою библиотеку). Код в ноутбуке проверяться не будет ❗❗❗

# ## Вариант 1 (простой, 10 баллов)
# 
# В данном задании вам потребуется сделать Python API для сервиса http://hollywood.mit.edu/GENSCAN.html
# 
# Он способен находить и вырезать интроны в переданной нуклеотидной последовательности. Делает он это не очень хорошо, но это лучше, чем ничего. К тому же у него действительно нет публичного API.
# 
# Реализуйте следующую функцию:
# `run_genscan(sequence=None, sequence_file=None, organism="Vertebrate", exon_cutoff=1.00, sequence_name="")` &mdash; выполняет запрос аналогичный заполнению формы на сайте. Принимает на вход все параметры, которые можно указать на сайте (кроме Print options). `sequence` &mdash; последовательность в виде строки или любого удобного вам типа данных, `sequence_file` &mdash; путь к файлу с последовательностью, который может быть загружен и использован вместо `sequence`. Функция должна будет возвращать объект типа `GenscanOutput`. Про него дальше.
# 
# Реализуйте **датакласс** `GenscanOutput`, у него должны быть следующие поля:
# + `status` &mdash; статус запроса
# + `cds_list` &mdash; список предсказанных белковых последовательностей с учётом сплайсинга (в самом конце результатов с сайта)
# + `intron_list` &mdash; список найденных интронов. Один интрон можно представить любым типом данных, но он должен хранить информацию о его порядковом номере, его начале и конце. Информацию о интронах можно получить из первой таблицы в результатах на сайте.
# + `exon_list` &mdash; всё аналогично интронам, но только с экзонами.
# 
# По желанию можно добавить любые данные, которые вы найдёте в результатах

# In[21]:


from genscan import run_genscan


# In[22]:


peptide='ffvskbvldbgfvldbglbkdfgkbdkfgb'


# In[23]:


output = run_genscan(sequence=peptide, organism="Vertebrate", 
                       exon_cutoff=1.00, sequence_name="")


# In[24]:


output.cds_list


# In[25]:


sequence_file = 'data/sequence.fasta'


# In[26]:


output = run_genscan(sequence_file=sequence_file, organism="Vertebrate", 
                       exon_cutoff=1.00, sequence_name="")


# In[27]:


output.cds_list


# In[28]:


output.intron_list


# ## Вариант 2 (очень сложный, 20 дополнительных баллов)

# В этом варианте от вас потребуется сделать Python API для BLAST, а именно для конкретной вариации **tblastn** https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=tblastn&PAGE_TYPE=BlastSearch&LINK_LOC=blasthome
# 
# Хоть у BLAST и есть десктопное приложение, всё-таки есть одна область, где API может быть полезен. Если мы хотим искать последовательность в полногеномных сборках (WGS), а не в базах данных отдельных генов, у нас могут возникнуть проблемы. Так как если мы хотим пробластить нашу последовательность против большого количества геномов нам пришлось бы или вручную отправлять запросы на сайте, или скачивать все геномы и делать поиск локально. И тот и другой способы не очень удобны, поэтому круто было бы иметь способ сделать автоматический запрос, не заходя в браузер.
# 
# Необходимо написать функцию для запроса, которая будет принимать 3 обязательных аргумента: **белковая последовательность**, которую мы бластим, **базу данных** (в этом задании нас интересует только WGS, но по желанию можете добавить какую-нибудь ещё), **таксон**, у которого мы ищем последовательность, чаще всего &mdash; конкретный вид. По=желанию можете добавить также любые другие аргументы, соответствующие различным настройкам поиска на сайте. 
# 
# Функция дожна возвращать список объектов типа `Alignment`, у него должны быть следующие атрибуты (всё согласно результатам в браузере, удобно посмотреть на рисунке ниже), можно добавить что-нибудь своё:
# 
# ![Alignment.png](attachment:e45d0969-ff95-4d4b-8bbc-7f5e481dcda3.png)
# 
# 
# Самое сложное в задании - правильно сделать запрос. Для этого нужно очень глубоко погрузиться в то, что происходит при отправке запроса при помощи инструмента для разработчиков. Ещё одна проблема заключается в том, что BLAST не отдаёт результаты сразу, какое-то время ваш запрос обрабатывается, при этом изначальный запрос не перекидывает вас на страницу с результатами. Задание не такое простое как кажется из описания!

# In[ ]:


# Не пиши код здесь, сделай отдельный модуль

