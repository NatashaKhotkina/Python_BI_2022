#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import seaborn as sns


# ## 1. Распарсите файл references

# In[3]:


pattern = r"ftp[\w\./]+"


# In[4]:


answer = []
with open('references.txt') as ref_file:
    for string in ref_file:
        answer += re.findall(pattern, string)


# In[5]:


len(answer)


# ## 2. Извлеките из рассказа  2430 A.D. все числа

# In[6]:


pattern = r"\d*\.?\d+"


# In[7]:


answer = []
with open('2430AD.txt') as ref_file:
    for string in ref_file:
        answer += re.findall(pattern, string)


# In[8]:


answer


# ## 3. Из того же рассказа извлеките все слова, в которых есть буква a

# In[9]:


pattern = r"\w*[aA]\w*"


# In[10]:


answer = []
with open('2430AD.txt') as ref_file:
    for string in ref_file:
        answer += re.findall(pattern, string)
answer


# ## Извлеките из рассказа все восклицательные предложения

# In[11]:


pattern = r"[A-Z][\w ,'\-]+!"
answer = []
with open('2430AD.txt') as ref_file:
    for string in ref_file:
        answer += re.findall(pattern, string, flags=re.ASCII)
answer


# ## Постройте гистограмму распределения длин уникальных слов

# In[12]:


pattern = r"[A-Za-z]+"
answer = []
with open('2430AD.txt') as ref_file:
    for string in ref_file:
        answer += re.findall(pattern, string, flags=re.ASCII)


# In[13]:


answer = list(map(lambda x: x.lower(), answer))
answer = list(set(answer))


# In[14]:


word_length = list(map(lambda x: len(x), answer))


# In[15]:


sns.set(rc={'figure.figsize':(20,15)})
sns.set(font_scale=2)
sns.histplot(word_length, bins=max(word_length))


# ## 6. Сделайте функцию-переводчик с русского на "кирпичный язык" 

# In[16]:


def brick_translator_rus(sentence):
    pattern = r"([ёуеыаоэяиюЁУЕЫАОЭЯИЮ])"
    return re.sub(pattern, r"\1к\1", sentence)


# In[17]:


brick_translator_rus("привет")


# ## 7. Сделайте функцию для извлечения из текста предложений с заданным количеством слов

# In[18]:


def find_n_words_sentences(sentense, n):
    pattern = r"[.!?] " + r"(\w+) " * (n - 1) + r"(\w+)(?=[.!?])"
    return re.findall(pattern, ". "+sentense+".")


# In[19]:


find_n_words_sentences("Здесь не три слова. Здесь тоже не три", 4)

