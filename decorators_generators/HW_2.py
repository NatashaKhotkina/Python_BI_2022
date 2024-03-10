#!/usr/bin/env python
# coding: utf-8

# # Задание 1 (2 балла)

# Напишите класс `MyDict`, который будет полностью повторять поведение обычного словаря, за исключением того, что при итерации мы должны получать и ключи, и значения.
# 
# **Модули использовать нельзя**

# In[2]:


class DictIterator:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.__current_iteration = 0
        self.length = len(dictionary)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.__current_iteration < self.length:
            items_iter = iter(self.dictionary.items())
            for i in range(self.__current_iteration):
                next(items_iter)
            result = next(items_iter)
        else:
            raise StopIteration
        self.__current_iteration += 1
        return result


class MyDict(dict):
    def __init__(self, dictionary):
        super().__init__(dictionary)
        self.dictionary = dictionary
        
    def __iter__(self):
        return DictIterator(self.dictionary)


# Или

# In[3]:


class MyDict(dict):        
    def __iter__(self):
        for items in self.items():
            yield items


# In[4]:


dct = MyDict({"a": 1, "b": 2, "c": 3, "d": 25})
for key, value in dct:
    print(key, value)   


# In[5]:


for key, value in dct.items():
    print(key, value)


# In[6]:


for key in dct.keys():
    print(key)


# In[7]:


dct["c"] + dct["d"]


# # Задание 2 (2 балла)

# Напишите функцию `iter_append`, которая "добавляет" новый элемент в конец итератора, возвращая итератор, который включает изначальные элементы и новый элемент. Итерироваться по итератору внутри функции нельзя, то есть вот такая штука не принимается
# ```python
# def iter_append(iterator, item):
#     lst = list(iterator) + [item]
#     return iter(lst)
# ```
# 
# **Модули использовать нельзя**

# In[8]:


def iter_append(iterator, item):
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            yield item
            break
    

my_iterator = iter([1, 2, 3])
new_iterator = iter_append(my_iterator, 4)

for element in new_iterator:
    print(element)


# # Задание 3 (5 баллов)

# Представим, что мы установили себе некотурую библиотеку, которая содержит в себе два класса `MyString` и `MySet`, которые являются наследниками `str` и `set`, но также несут и дополнительные методы.
# 
# Проблема заключается в том, что библиотеку писали не очень аккуратные люди, поэтому получилось так, что некоторые методы возвращают не тот тип данных, который мы ожидаем. Например, `MyString().reverse()` возвращает объект класса `str`, хотя логичнее было бы ожидать объект класса `MyString`.
# 
# Найдите и реализуйте удобный способ сделать так, чтобы подобные методы возвращали экземпляр текущего класса, а не родительского. При этом **код методов изменять нельзя**
# 
# **+3 дополнительных балла** за реализацию того, чтобы **унаследованные от `str` и `set` методы** также возвращали объект интересующего нас класса (то есть `MyString.replace(..., ...)` должен возвращать `MyString`). **Переопределять методы нельзя**
# 
# **Модули использовать нельзя**

# In[9]:


# Ваш код где угодно, но не внутри методов
def self_func(func):
    def inner_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if isinstance(result, str) or isinstance(result, set):
            result = type(self)(result)
        return result
    return inner_func
    


def self_class(cls):
    methods = [method for method in dir(cls) if callable(getattr(cls, method)) and
                                     not isinstance(getattr(cls, method), type) and 
                                     method != '__new__' and method != '__init__']
    for method_name in methods:
        func = getattr(cls, method_name)
        decorated_func = self_func(func)
        setattr(cls, method_name, decorated_func)
    return cls
        
        
@self_class
class MyString(str):
    def reverse(self):
        return self[::-1]
    
    def make_uppercase(self):
        return "".join([chr(ord(char) - 32) if 97 <= ord(char) <= 122 else char for char in self])
    
    def make_lowercase(self):
        return "".join([chr(ord(char) + 32) if 65 <= ord(char) <= 90 else char for char in self])
    
    def capitalize_words(self):
        return " ".join([word.capitalize() for word in self.split()])
    

@self_class
class MySet(set):
    def is_empty(self):
        return len(self) == 0
    
    def has_duplicates(self):
        return len(self) != len(set(self))
    
    def union_with(self, other):
        return self.union(other)
    
    def intersection_with(self, other):
        return self.intersection(other)
    
    def difference_with(self, other):
        return self.difference(other)


# In[10]:


string_example = MyString("Aa Bb Cc")
set_example_1 = MySet({1, 2, 3, 4})
set_example_2 = MySet({3, 4, 5, 6, 6})

print(type(string_example.reverse()))
print(type(string_example.make_uppercase()))
print(type(string_example.make_lowercase()))
print(type(string_example.capitalize_words()))
print()
print(type(set_example_1.is_empty()))
print(type(set_example_2.has_duplicates()))
print(type(set_example_1.union_with(set_example_2)))
print(type(set_example_1.difference_with(set_example_2)))


# # Задание 4 (5 баллов)

# Напишите декоратор `switch_privacy`:
# 1. Делает все публичные **методы** класса приватными
# 2. Делает все приватные методы класса публичными
# 3. Dunder методы и защищённые методы остаются без изменений
# 4. Должен работать тестовый код ниже, в теле класса писать код нельзя
# 
# **Модули использовать нельзя**

# In[11]:


def decoratorfunc(cls):
    private_start = '_' + cls.__name__ + '__'
    for method_name in dir(cls):
        func = getattr(cls, method_name)
        if callable(func):
            if method_name.startswith(private_start):
                setattr(cls, method_name.replace(private_start, ''), func)
                delattr(cls, method_name)
            elif (not method_name.startswith('_')) and (not method_name.endswith('_')):
                setattr(cls, private_start + method_name, func)
                delattr(cls, method_name)
    return cls


# In[12]:


# Ваш код здесь
@decoratorfunc
class ExampleClass:
    # Но не здесь
    def public_method(self):
        return 1
    
    def _protected_method(self):
        return 2
    
    def __private_method(self):
        return 3
    
    def __dunder_method__(self):
        pass


# In[13]:


test_object = ExampleClass()

test_object._ExampleClass__public_method()   # Публичный метод стал приватным


# In[14]:


test_object.private_method()   # Приватный метод стал публичным


# In[15]:


test_object._protected_method()   # Защищённый метод остался защищённым


# In[16]:


test_object.__dunder_method__()   # Дандер метод не изменился


# In[17]:


hasattr(test_object, "public_method"), hasattr(test_object, "private")   # Изначальные варианты изменённых методов не сохраняются


# # Задание 5 (7 баллов)

# Напишите [контекстный менеджер](https://docs.python.org/3/library/stdtypes.html#context-manager-types) `OpenFasta`
# 
# Контекстные менеджеры это специальные объекты, которые могут работать с конструкцией `with ... as ...:`. В них нет ничего сложного, для их реализации как обычно нужно только определить только пару dunder методов. Изучите этот вопрос самостоятельно
# 
# 1. Объект должен работать как обычные файлы в питоне (наследоваться не надо, здесь лучше будет использовать **композицию**), но:
#     + При итерации по объекту мы должны будем получать не строку из файла, а специальный объект `FastaRecord`. Он будет хранить в себе информацию о последовательности. Важно, **не строки, а именно последовательности**, в fasta файлах последовательность часто разбивают на много строк
#     + Нужно написать методы `read_record` и `read_records`, которые по смыслу соответствуют `readline()` и `readlines()` в обычных файлах, но они должны выдавать не строки, а объект(ы) `FastaRecord`
# 2. Конструктор должен принимать один аргумент - **путь к файлу**
# 3. Класс должен эффективно распоряжаться памятью, с расчётом на работу с очень большими файлами
#     
# Объект `FastaRecord`. Это должен быть **датакласс** (см. про примеры декораторов в соответствующей лекции) с тремя полями:
# + `seq` - последовательность
# + `id_` - ID последовательности (это то, что в фаста файле в строке, которая начинается с `>` до первого пробела. Например, >**GTD326487.1** Species anonymous 24 chromosome) 
# + `description` - то, что осталось после ID (Например, >GTD326487.1 **Species anonymous 24 chromosome**)
# 
# 
# Напишите демонстрацию работы кода с использованием всех написанных методов, обязательно добавьте файл с тестовыми данными в репозиторий (не обязательно большой)
# 
# **Можно использовать модули из стандартной библиотеки**

# In[18]:


import os
from dataclasses import dataclass


# In[19]:


@dataclass
class FastaRecord:
    seq: str
    id_: str
    description: str


# In[20]:


class OpenFasta:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.id_descr = None
        self.end = False
         
    def __enter__(self):
        self.file = open(self.filename)
        return self
     
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file.close()
        
    def read_record(self):
        if self.id_descr is None:
            self.id_descr = self.file.readline().rstrip()
        if self.id_descr == '':
            return FastaRecord(seq=None, id_=None, description=None)
  
        if ' ' in self.id_descr:
            id_ , description = self.id_descr.split(' ', 1)
        else:
            id_ = self.id_descr
            description = None
        id_ = id_[1:]
        new_lines = []
        
        while True:
            new_line = self.file.readline().rstrip()
            if new_line.startswith('>'):
                self.id_descr = new_line
                break
            elif new_line == '':
                self.end = True
                break
            else:
                new_lines.append(new_line)
        seq = ''.join(new_lines)

        return FastaRecord(seq=seq, id_=id_, description=description)
    
    def read_records(self):
        records = []
        while not self.end:
            records.append(self.read_record())
        return records
    
    def __next__(self):
        if self.end:
            raise StopIteration
        else:
            return self.read_record()
    
    def __iter__(self):
        return self


# In[21]:


with OpenFasta(os.path.join("data", "fasta_file.fasta")) as fasta:
    print(fasta.read_record())
    print("let's continue")
    print(fasta.read_records())


# In[22]:


with OpenFasta(os.path.join("data", "fasta_file.fasta")) as fasta:
    for seq in fasta:
        print(seq)


# # Задание 6 (7 баллов)

# 1. Напишите код, который позволит получать все возможные (неуникальные) генотипы при скрещивании двух организмов. Это может быть функция или класс, что вам кажется более удобным.
# 
# Например, все возможные исходы скрещивания "Aabb" и "Aabb" (неуникальные) это
# 
# ```
# AAbb
# AAbb
# AAbb
# AAbb
# Aabb
# Aabb
# Aabb
# Aabb
# Aabb
# Aabb
# Aabb
# Aabb
# aabb
# aabb
# aabb
# aabb
# ```
# 
# 2. Напишите функцию, которая вычисляет вероятность появления определённого генотипа (его ожидаемую долю в потомстве).
# Например,
# 
# ```python
# get_offspting_genotype_probability(parent1="Aabb", parent2="Aabb", target_genotype="Аabb")   # 0.5
# 
# ```
# 
# 3. Напишите код, который выводит все уникальные генотипы при скрещивании `'АаБбввГгДдЕеЖжЗзИиЙйккЛлМмНн'` и `'АаббВвГгДДЕеЖжЗзИиЙйКкЛлМмНН'`, которые содержат в себе следующую комбинацию аллелей `'АаБбВвГгДдЕеЖжЗзИиЙйКкЛл'`
# 4. Напишите код, который расчитывает вероятность появления генотипа `'АаБбввГгДдЕеЖжЗзИиЙйккЛлМмНн'` при скрещивании `АаБбВвГгДдЕеЖжЗзИиЙйКкЛлМмНн` и `АаБбВвГгДдЕеЖжЗзИиЙйКкЛлМмНн`
# 
# Важные замечания:
# 1. Порядок следования аллелей в случае гетерозигот всегда должен быть следующим: сначала большая буква, затем маленькая (вариант `AaBb` допустим, но `aAbB` быть не должно)
# 2. Подзадачи 3 и 4 могут потребовать много вычислительного времени (до 15+ минут в зависимости от железа), поэтому убедитесь, что вы хорошо протестировали написанный вами код на малых данных перед выполнением этих задач. Если ваш код работает **дольше 20 мин**, то скорее всего ваше решение не оптимально, попытайтесь что-нибудь оптимизировать. Если оптимальное решение совсем не получается, то попробуйте из входных данных во всех заданиях убрать последний ген (это должно уменьшить время выполнения примерно в 4 раза), но **за такое решение будет снято 2 балла**
# 3. Несмотря на то, что подзадания 2, 3 и 4 возможно решить математически, не прибегая к непосредственному получению всех возможных генотипов, от вас требуется именно brute-force вариант алгоритма
# 
# **Можно использовать модули из стандартной библиотеки питона**, но **за выполнение задания без использования модулей придусмотрено +3 дополнительных балла**

# In[18]:


# Ваш код здесь (1 и 2 подзадание)


# In[20]:


# Ваш код здесь (3 подзадание)


# In[21]:


# Ваш код здесь (4 подзадание)

