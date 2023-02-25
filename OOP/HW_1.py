#!/usr/bin/env python
# coding: utf-8

# # Задание 1 (5 баллов)

# Напишите классы **Chat**, **Message** и **User**. Они должны соответствовать следующим требованиям:
# 
# **Chat**:
# + Должен иметь атрибут `chat_history`, где будут храниться все сообщения (`Message`) в обратном хронологическом порядке (сначала новые, затем старые)
# + Должен иметь метод `show_last_message`, выводящий на экран информацию о последнем сообщении
# + Должен иметь метод `get_history_from_time_period`, который принимает два опциональных аргумента (даты с которой и по какую мы ищем сообщения и выдаём их). Метод также должен возвращать объект типа `Chat`
# + Должен иметь метод `show_chat`, выводящий на экран все сообщения (каждое сообщение в таком же виде как и `show_last_message`, но с разделителем между ними)
# + Должен иметь метод `recieve`, который будет принимать сообщение и добавлять его в чат
# 
# **Message**:
# + Должен иметь три обязательных атрибута
#     + `text` - текст сообщения
#     + `datetime` - дата и время сообщения (встроенный модуль datetime вам в помощь). Важно! Это должна быть не дата создания сообщения, а дата его попадания в чат! 
#     + `user` - информация о пользователе, который оставил сообщение (какой тип данных использовать здесь, разберётесь сами)
# + Должен иметь метод `show`, который печатает или возвращает информацию о сообщении с необходимой информацией (дата, время, юзер, текст)
# + Должен иметь метод `send`, который будет отправлять сообщение в чат
# 
# **User**:
# + Класс с информацией о юзере, наполнение для этого класса придумайте сами
# 
# Напишите несколько примеров использования кода, которое показывает взаимодействие между объектами.
# 
# В тексте задания намерено не указано, какие аргументы должны принимать методы, пускай вам в этом поможет здравый смысл)
# 
# В этом задании не стоит флексить всякими продвинутыми штуками, для этого есть последующие
# 
# В этом задании можно использовать только модуль `datetime`

# In[1]:


from datetime import datetime


# In[2]:


class Chat:
    def __init__(self, history=None):
        if history is None:
            history = []
        self.chat_history = history
        
    def show_last_message(self):
        print(self.chat_history[0].user, self.chat_history[0].text, 
              self.chat_history[0].datetime, sep='\n')
    
    def get_history_from_time_period(self, begin=None, end=None):
        if begin is None:
            begin = self.chat_history[-1].datetime
        if end is None:
            end = self.chat_history[0].datetime
        history = list(mes for mes in self.chat_history if mes.datetime >= begin and mes.datetime <= end)
        return Chat(history)
    
    def show_chat(self):
        for mes in self.chat_history[::-1]:
            print(mes.user, mes.text, mes.datetime, sep='\n', end='\n❤️️❤️️❤️️\n')
            
    
    def receive(self, message):
        self.chat_history = [message] + self.chat_history


class Message:
    def __init__(self, text, user):
        self.text = text
        self.datetime = None
        self.user = user.nickname
    
    def show(self):
        print(self.user, self.text, self.datetime, sep='\n')
        
    def send(self, chat):
        self.datetime = datetime.now()
        chat.receive(self)
        

class User:
    def __init__(self, name, surname, nickname):
        self.name = name
        self.surname = surname
        self.nickname = nickname


# In[3]:


roma = User('Roma', 'Kruglikov', 'sunny')
natasha = User('Natasha', 'Khotkina', 'crying')


# In[4]:


mes1 = Message('Hi, Natasha!', roma)
mes2 = Message('Hi!', natasha)


# In[5]:


mes1.show()


# In[6]:


friends_chat = Chat()


# In[7]:


mes1.send(friends_chat)


# In[8]:


mes2.send(friends_chat)


# In[9]:


friends_chat.show_chat()


# In[12]:


chat_hist = friends_chat.get_history_from_time_period(begin=datetime(2023,2,25,18,46,59))


# In[13]:


chat_hist.show_chat()


# # Задание 2 (3 балла)

# В питоне как-то слишком типично и неинтересно происходят вызовы функций. Напишите класс `Args`, который будет хранить в себе аргументы, а функции можно будет вызывать при помощи следующего синтаксиса.
# 
# Использовать любые модули **нельзя**, да и вряд-ли это как-то поможет)

# In[14]:


class Args():
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def __rlshift__(self, func):
        print(func(*self.args, **self.kwargs))


# In[15]:


sum << Args([1, 2])


# In[16]:


(lambda a, b, c: a**2 + b + c) << Args(1, 2, c=50)


# # Задание 3 (5 баллов)

# Сделайте класс наследник `float`. Он должен вести себя как `float`, но также должен обладать некоторыми особенностями:
# + При получении атрибутов формата `<действие>_<число>` мы получаем результат такого действия над нашим числом
# + Создавать данные атрибуты в явном виде, очевидно, не стоит
# 
# Подсказка: если в процессе гуглёжки, вы выйдете на такую тему как **"Дескрипторы", то это НЕ то, что вам сейчас нужно**
# 
# Примеры использования ниже

# In[17]:


class StrangeFloat(float):
    def __getattribute__(self, name):
        if hasattr(float, name):
            return super().__getattribute__(name)
        else:
            operation, number = name.split('_')
            number = float(number)
            operation_dict = {
                'add': self.__add__,
                'subtract': self.__sub__, 
                'multiply': self.__mul__,
                'divide': self.__truediv__
            }
            return StrangeFloat(operation_dict[operation](number))
            


# In[18]:


number = StrangeFloat(3.5)


# In[19]:


number.add_1


# In[20]:


number.subtract_20


# In[21]:


number.multiply_5


# In[22]:


number.divide_25


# In[23]:


number.add_1.add_2.multiply_6.divide_8.subtract_9


# In[24]:


getattr(number, "add_-2.5")   # Используем getattr, так как не можем написать number.add_-2.5 - это SyntaxError


# In[25]:


number + 8   # Стандартные для float операции работают также


# In[26]:


number.as_integer_ratio()   # Стандартные для float операции работают также  (это встроенный метод float, писать его НЕ НАДО)


# # Задание 4 (3 балла)

# В данном задании мы немного отдохнём и повеселимся. От вас требуется заменить в данном коде максимально возможное количество синтаксических конструкций на вызовы dunder методов, dunder атрибутов и dunder переменных.
# 
# Маленькая заметка: полностью всё заменить невозможно. Например, `function()` можно записать как `function.__call__()`, но при этом мы всё ещё не избавляемся от скобочек, так что можно делать так до бесконечности `function.__call__.__call__.__call__.__call__.....__call__()` и при всём при этом мы ещё не избавляемся от `.` для доступа к атрибутам. В общем, замените всё, что получится, не закапываясь в повторы, как в приведённом примере. Чем больше разных методов вы найдёте и используете, тем лучше и тем выше будет балл
# 
# Код по итогу дожен работать и печатать число **4420.0**, как в примере. Структуру кода менять нельзя, просто изменяем конструкции на синонимичные
# 
# И ещё маленькая подсказка. Заменить здесь можно всё кроме:
# + Конструкции `for ... in ...`:
# + Синтаксиса создания лямбда функции
# + Оператора присваивания `=`
# + Конструкции `if-else`

# In[35]:


np = __import__('numpy')


matrix = []
for idx in range(0, 100, 10):
    matrix.__iadd__([list(range(idx, idx.__add__(10)))])
    
selected_columns_indices = list(filter(lambda x: x in range(1, 5, 2), range(matrix.__len__())))
selected_columns = map(lambda x: [x.__getitem__(col) for col in selected_columns_indices], matrix)

arr = np.array(list(selected_columns))

mask = (np.array(list(i.__getitem__(1) for i in arr)).__mod__(3)).__eq__(0)
new_arr = arr.__getitem__(mask)

product = new_arr.__matmul__(new_arr.T)

if (product.__getitem__(0).__lt__(1000)).all() and (product.__getitem__(2).__gt__(1000)).any():
    print(product.mean())


# # Задание 5 (10 баллов)

# Напишите абстрактный класс `BiologicalSequence`, который задаёт следующий интерфейс:
# + Работа с функцией `len`
# + Возможность получать элементы по индексу и делать срезы последовательности (аналогично строкам)
# + Вывод на печать в удобном виде и возможность конвертации в строку
# + Возможность проверить алфавит последовательности на корректность
# 
# Напишите класс `NucleicAcidSequence`:
# + Данный класс реализует интерфейс `BiologicalSequence`
# + Данный класс имеет новый метод `complement`, возвращающий комплементарную последовательность
# + Данный класс имеет новый метод `gc_content`, возвращающий GC-состав (без разницы, в процентах или в долях)
# 
# Напишите классы наследники `NucleicAcidSequence`: `DNASequence` и `RNASequence`
# + `DNASequence` должен иметь метод `transcribe`, возвращающий транскрибированную РНК-последовательность
# + Данные классы не должны иметь <ins>публичных методов</ins> `complement` и метода для проверки алфавита, так как они уже должны быть реализованы в `NucleicAcidSequence`.
# 
# Напишите класс `AminoAcidSequence`:
# + Данный класс реализует интерфейс `BiologicalSequence`
# + Добавьте этому классу один любой метод, подходящий по смыслу к аминокислотной последовательности. Например, метод для нахождения изоэлектрической точки, молекулярного веса и т.д.
# 
# Комментарий по поводу метода `NucleicAcidSequence.complement`, так как я хочу, чтобы вы сделали его опредедённым образом:
# 
# При вызове `dna.complement()` или условного `dna.check_alphabet()` должны будут вызываться соответствующие методы из `NucleicAcidSequence`. При этом, данный метод должен обладать свойством полиморфизма, иначе говоря, внутри `complement` не надо делать условия а-ля `if seuqence_type == "DNA": return self.complement_dna()`, это крайне не гибко. Данный метод должен опираться на какой-то общий интерфейс между ДНК и РНК. Создание экземпляров `NucleicAcidSequence` не подразумевается, поэтому код `NucleicAcidSequence("ATGC").complement()` не обязан работать, а в идеале должен кидать исключение `NotImplementedError` при вызове от экземпляра `NucleicAcidSequence`
# 
# Вся сложность задания в том, чтобы правильно организовать код. Если у вас есть повторяющийся код в сестринских классах или родительском и дочернем, значит вы что-то делаете не так.
# 
# 
# Маленькое замечание: По-хорошему, между классом `BiologicalSequence` и классами `NucleicAcidSequence` и `AminoAcidSequence`, ещё должен быть класс-прослойка, частично реализующий интерфейс `BiologicalSequence`, но его писать не обязательно, так как задание и так довольно большое (правда из-за этого у вас неминуемо возникнет повторяющийся код в классах `NucleicAcidSequence` и `AminoAcidSequence`)

# In[36]:


from abc import ABC, abstractmethod


class BiologicalSequence(ABC):
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, slc):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass
    
    @abstractmethod
    def check_alphabet(self):
        pass
    

class BiologicalSequence_impl(BiologicalSequence):
    def __init__(self, seq):
        self.seq = seq
    
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, slc):
        return self.seq.__getitem__(slc)
    
    def __repr__(self):
        return str(self.seq)
    
    def check_alphabet(self):
        for letter in self.seq:
            if letter.upper() not in self.alphabet:
                return False
        return True
    
    
class NucleicAcidSequence(BiologicalSequence_impl):
    
    def complement(self, inplace=False):
        complement_seq = self.seq.translate(self.trans_object)
        if inplace:
            self.seq = complement_seq
        return complement_seq
    
    def gc_content(self):
        return (self.seq.upper().count('G') + self.seq.upper().count('C')) / len(self.seq)
    

class DNASequence(NucleicAcidSequence):
    def __init__(self, seq):
        super().__init__(seq)
        self.alphabet = 'ATGC'
        self.trans_object = str.maketrans("ATGCatgc", "TACGtacg")
        
    def transcribe(self):
        transcribed_seq = self.seq.translate(str.maketrans("ATGCatgc", "UACGuacg"))
        return transcribed_seq
        

class RNASequence(NucleicAcidSequence):
    def __init__(self, seq):
        super().__init__(seq)
        self.alphabet = 'AUGC'
        self.trans_object = str.maketrans("AGCUagcu", "UCGAucga")

        
class AminoAcidSequence(BiologicalSequence_impl):
    def __init__(self, seq):
        super().__init__(seq)
        self.alphabet = 'ACDEFGHIKLMNPQRSTVWXY'
    
    def one_to_three_letters(self):
        three_letter ={'V':'VAL', 'I':'ILE', 'L':'LEU', 'E':'GLU', 'Q':'GLN', \
                        'D':'ASP', 'N':'ASN', 'H':'HIS', 'W':'TRP', 'F':'PHE', 'Y':'TYR',    \
                        'R':'ARG', 'K':'LYS', 'S':'SER', 'T':'THR', 'M':'MET', 'A':'ALA',    \
                        'G':'GLY', 'P':'PRO', 'C':'CYS'}
        three_letter_seq = self.seq.translate(str.maketrans(three_letter))
        return three_letter_seq


# In[37]:


dna1 = DNASequence('ATGGGGC')


# In[38]:


dna1.check_alphabet()


# In[39]:


dna1.transcribe()


# In[40]:


dna1.complement()


# In[41]:


dna1.gc_content()


# In[42]:


len(dna1)


# In[43]:


rna1 = RNASequence('ATGGGGC')


# In[44]:


rna1.check_alphabet()


# In[45]:


rna2 = RNASequence('AUGGC')


# In[46]:


rna2.check_alphabet()


# In[47]:


rna2.complement()


# In[48]:


rna2.gc_content()


# In[49]:


len(rna2)


# In[50]:


rna2[0]


# In[51]:


rna2[-1]


# In[52]:


rna2[1:]


# In[53]:


rna2


# In[54]:


str(rna2)


# In[55]:


am_ac = AminoAcidSequence('VDRGLHSC')


# In[56]:


am_ac.one_to_three_letters()


# In[48]:


am_ac.check_alphabet()


# In[50]:


am_ac[3:7]

