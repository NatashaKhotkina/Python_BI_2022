#!/usr/bin/env python
# coding: utf-8

# В формулировке заданий будет использоваться понятие **worker**. Это слово обозначает какую-то единицу параллельного выполнения, в случае питона это может быть **поток** или **процесс**, выбирайте то, что лучше будет подходить к конкретной задаче
# 
# В каждом задании нужно писать подробные аннотиции типов для:
# 1. Аргументов функций и классов
# 2. Возвращаемых значений
# 3. Классовых атрибутов (если такие есть)
# 
# В каждом задании нужно писать докстроки в определённом стиле (какой вам больше нравится) для всех функций, классов и методов

# In[1]:


from typing import Union, Callable, Any


# # Задание 1 (7 баллов)

# В одном из заданий по ML от вас требовалось написать кастомную реализацию Random Forest. Её проблема состоит в том, что она работает медленно, так как использует всего один поток для работы. Добавление параллельного программирования в код позволит получить существенный прирост в скорости обучения и предсказаний.
# 
# В данном задании от вас требуется добавить возможность обучать случайный лес параллельно и использовать параллелизм для предсказаний. Для этого вам понадобится:
# 1. Добавить аргумент `n_jobs` в метод `fit`. `n_jobs` показывает количество worker'ов, используемых для распараллеливания
# 2. Добавить аргумент `n_jobs` в методы `predict` и `predict_proba`
# 3. Реализовать функционал по распараллеливанию в данных методах
# 
# В результате код `random_forest.fit(X, y, n_jobs=2)` и `random_forest.predict(X, y, n_jobs=2)` должен работать в ~1.5-2 раза быстрее, чем `random_forest.fit(X, y, n_jobs=1)` и `random_forest.predict(X, y, n_jobs=1)` соответственно
# 
# Если у вас по каким-то причинам нет кода случайного леса из ДЗ по ML, то вы можете написать его заново или попросить у однокурсника. *Детали* реализации ML части оцениваться не будут, НО, если вы поломаете логику работы алгоритма во время реализации параллелизма, то за это будут сниматься баллы
# 
# В задании можно использовать только модули из **стандартной библиотеки** питона, а также функции и классы из **sklearn** при помощи которых вы изначально писали лес

# In[3]:


from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import random
import numpy as np


# In[4]:


SEED = 111


# In[24]:


class RandomForestClassifierCustom(BaseEstimator):
    """
    Custom random forest classifier is a composition of sklearn DecisionTreeClassifiers. 

    Attributes:
        n_estimators: Number of DecisionTreeClassifiers to compose the Random Forest.
        max_depth: Maximum depth of each DecisionTreeClassifier.
        max_features: The number of features to consider when looking for the best split
        random_state: Random state. The actual random state which is passed to each DecisionTreeClassifier 
        is counted from this attribute.
        trees: List of fitted trees. The list is filled after fit method is called.
        feat_ids_by_tree: Indices of columns for each tree. (In RandomForestClassifier each tree 'sees' 
        only a subset of columns).
        classes_: Number of classes in train data.
    """
    def __init__(
        self, n_estimators:int = 10, max_depth:int = None, max_features:int = None, random_state:int = SEED
    ) -> None:
        """
        Initialization function.

        Args:
            n_estimators: Number of DecisionTreeClassifiers to compose the Random Forest.
            max_depth: Maximum depth of each DecisionTreeClassifier.
            max_features: The number of features to consider when looking for the best split
            random_state: Random state. The actual random state which is passed to each DecisionTreeClassifier 

        Returns:
            None.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        self.feat_ids_by_tree = []

    def fit_some_trees(self, arg_tuple: tuple[np.ndarray, np.ndarray, int]) -> tuple:
        """
        Fits a subset of trees on subset of data passed to it.

        Args:
            arg_tuple: Tuple of (X, y, n_estimators_work). X and y are a subset of training data and 
            n_estimators_work is the number of trees to train.

        Returns:
            List of trained trees and list of column indices for each tree.
        """
        X, y, n_estimators_work = arg_tuple
        some_trees = []
        feat_ids_by_tree = []
        for i in range(n_estimators_work):
            num_rows, num_columns = X.shape
            random_state = SEED
            random.seed(random_state)
            np.random.seed(random_state)

            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features, 
                                          random_state=random_state)
            subset_col_idx = np.random.choice(num_columns, self.max_features, replace=False)
            subset_row_idx = np.random.choice(num_rows, num_rows // 2, replace=True)
            X_subset = X[subset_row_idx, :][:, subset_col_idx]
            y_subset = y[subset_row_idx]

            tree.fit(X_subset, y_subset)
            
            some_trees.append(tree)
            feat_ids_by_tree.append(subset_col_idx)
        return some_trees, feat_ids_by_tree
                
    def fit(self, X: np.ndarray, y: np.ndarray, n_jobs: int = 1) -> RandomForestClassifierCustom:
        """
        Fits all trees creating n_jobs processes.

        Args:
            X: Features to predict on.
            y: y true; the correct classes of objects.
            n_jobs: Number of processes for parallel tree fitting.

        Returns:
            Self.
        """
        self.classes_ = sorted(np.unique(y))
        X = np.array(X)
        n_estimators = [self.n_estimators // n_jobs] * n_jobs
        n_estimators[-1] += self.n_estimators - sum(n_estimators)
        
        with ProcessPoolExecutor(n_jobs) as pool:                               
            result = pool.map(self.fit_some_trees, [(X, y, n) 
                                                    for n in n_estimators])
            for trees, feat_ids_by_tree in result:
                self.trees.extend(trees)
                self.feat_ids_by_tree.extend(feat_ids_by_tree)
        
        return self

    def predict_proba_some_trees(self, arg_tuple: tuple[np.ndarray, int, int]) -> list:
        """
        Predicts probability of each class with trees passed to the method.

        Args:
            arg_tuple: Tuple of (X, start, stop). X is subset of features to predict on; start and stop are 
            the indices of tree classifiers in self.trees.

        Returns:
            List of probability for each class predicted by each tree.
        """
        X, start, stop = arg_tuple
        proba = []
        for i in range(start, stop):
            tree = self.trees[i]
            subset_col_idx = self.feat_ids_by_tree[i]
            X_subset = X[:, subset_col_idx]
            proba.append(tree.predict_proba(X_subset))
        return proba
    
    
    def predict_proba(self, X: np.ndarray, n_jobs: int = 1) -> list:
        """
        Predicts probability of each class with all trees creating n_jobs processes.

        Args:
            X: Features to predict on.
            n_jobs: Number of processes for parallel predicting.

        Returns:
            Self.
        """
        indices = [self.n_estimators // n_jobs] * n_jobs
        indices[-1] += self.n_estimators - sum(indices)
        for i in range(1, len(indices)):
            indices[i] += indices[i - 1]
        indices = [0] + indices
        
        proba = []
        
        with ProcessPoolExecutor(n_jobs) as pool:                               
            result = pool.map(self.predict_proba_some_trees, [(X, indices[i], indices[i + 1]) 
                                                             for i in range(n_jobs)])
        for p in result:
            proba.extend(p)

        proba = np.array(proba)
        proba = np.mean(proba, axis=0)
        return proba
    
    def predict(self, X: np.ndarray, n_jobs: int = 1) -> np.ndarray:
        """
        Predicts the correct class.

        Args:
            X: Features to predict on.
            n_jobs: Number of processes for parallel predicting.

        Returns:
            Predictions of correct classes.
        """
        probas = self.predict_proba(X, n_jobs)
        predictions = np.argmax(probas, axis=1)
        
        return predictions


# In[25]:


X, y = make_classification(n_samples=100000)


# In[26]:


random_forest = RandomForestClassifierCustom(max_depth=30, n_estimators=100, max_features=2, random_state=42)


# In[27]:


get_ipython().run_cell_magic('time', '', '\n_ = random_forest.fit(X, y, n_jobs=1)\n')


# In[28]:


get_ipython().run_cell_magic('time', '', '\npreds_1 = random_forest.predict(X, n_jobs=1)\n')


# In[29]:


random_forest = RandomForestClassifierCustom(max_depth=30, n_estimators=100, max_features=2, random_state=42)


# In[30]:


get_ipython().run_cell_magic('time', '', '\n_ = random_forest.fit(X, y, n_jobs=2)\n')


# In[31]:


get_ipython().run_cell_magic('time', '', '\npreds_2 = random_forest.predict(X, n_jobs=2)\n')


# In[32]:


(preds_1 == preds_2).all()   # Количество worker'ов не должно влиять на предсказания


# #### Какие есть недостатки у вашей реализации параллельного Random Forest (если они есть)? Как это можно исправить? Опишите словами, можно без кода (+1 дополнительный балл)

# 1. У меня не получилось нормально передавать random seed, поскольку в каждом процессе свой цикл.
# 2. Не произошло действительно кратного ускорения. Процессы выполняют одинаковую работу, поэтому, видимо, не удается их удачно распараллелить

# # Задание 2 (9 баллов)

# Напишите декоратор `memory_limit`, который позволит ограничивать использование памяти декорируемой функцией.
# 
# Декоратор должен принимать следующие аргументы:
# 1. `soft_limit` - "мягкий" лимит использования памяти. При превышении функцией этого лимита должен будет отображён **warning**
# 2. `hard_limit` - "жёсткий" лимит использования памяти. При превышении функцией этого лимита должно будет брошено исключение, а функция должна немедленно завершить свою работу
# 3. `poll_interval` - интервал времени (в секундах) между проверками использования памяти
# 
# Требования:
# 1. Потребление функцией памяти должно отслеживаться **во время выполнения функции**, а не после её завершения
# 2. **warning** при превышении `soft_limit` должен отображаться один раз, даже если функция переходила через этот лимит несколько раз
# 3. Если задать `soft_limit` или `hard_limit` как `None`, то соответствующий лимит должен быть отключён
# 4. Лимиты должны передаваться и отображаться в формате `<number>X`, где `X` - символ, обозначающий порядок единицы измерения памяти ("B", "K", "M", "G", "T", ...)
# 5. В тексте warning'ов и исключений должен быть указан текщий объём используемой памяти и величина превышенного лимита
# 
# В задании можно использовать только модули из **стандартной библиотеки** питона, можно писать вспомогательные функции и/или классы
# 
# В коде ниже для вас предопределены некоторые полезные функции, вы можете ими пользоваться, а можете не пользоваться

# In[33]:


import os
import psutil
import time
import warnings
import threading


def get_memory_usage() -> int:    # Показывает текущее потребление памяти процессом
    """
    Shows memory occupied by a process.

    Args:
        None.

    Returns:
        Memory occupied by a process.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def bytes_to_human_readable(n_bytes: int) -> str:
    """
    Converts number of bytes to human-readable representation (such as 2K, 5M, etc).

    Args:
        n_bytes: Number of bytes to convert to human-readable representation.

    Returns:
        Human-readable representation of bytes number.
    """
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for idx, s in enumerate(symbols):
        prefix[s] = 1 << (idx + 1) * 10
    for s in reversed(symbols):
        if n_bytes >= prefix[s]:
            value = float(n_bytes) / prefix[s]
            return f"{value:.2f}{s}"
    return f"{n_bytes}B"

def human_readable_to_bytes(number: str) -> float:
    """
    Converts human-readable representation (such as 2K, 5M, etc) to number of bytes.

    Args:
        number: Human-readable representation of bytes number (such as 2K, 5M, etc).

    Returns:
        Number of bytes.
    """
    prefix = {'B': 0,'K': 1, 'M': 2, 'G': 3, 'T': 4, 'P': 5, 'E': 6, 'Z': 7, 'Y': 8}
    return float(number[:len(number) -1]) * (1024 ** prefix[number[-1]])


# In[36]:


class ThreadWithReturnValue(threading.Thread):
    """
    Custom class for creating threads which saves the result of function in thread to 'value' attribute. 

    Attributes:
        target: Function to execute in a thread.
        args: Args to pass to executable function ('target').
        kwargs: Kwargs to pass to executable function ('target').
        value: Value that is returned by 'target'.
    """
    def __init__(self, target: Callable, args: Any, kwargs: Any) -> None:
        """
        Initialization function.

        Args:
            target: Function to execute in a thread.
            args: Args to pass to executable function ('target').
            kwargs: Kwargs to pass to executable function ('target').

        Returns:
            None.
        """
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.value = None
    def run(self) -> None:
        """
        Runs the 'target' function in the thread and writed the result, returned by the 'target' function 
        to 'value' attribute.

        Args:
            None.

        Returns:
            None.
        """
        self.value = self.target(*self.args, **self.kwargs)


# In[37]:


def memory_limit(soft_limit:str = None, hard_limit:str = None, poll_interval: Union[int, float] = 1) -> Callable:
    """
    Returns decorator function which throws warning after exceeding the soft limit and stops execution 
    after exceeding the hard limit.

    Args:
        soft_limit: Memory limit after which a warning is shown.
        hard_limit: Memory limit after which execution is stopped.
        poll_interval: Time interval to recheck memory occupation.

    Returns:
        Decorator function.
    """
    def decorator(func):
        def inner_func(*args, **kwargs):
            nonlocal soft_limit, hard_limit
            if soft_limit:
                soft_limit_b = human_readable_to_bytes(soft_limit)
            if hard_limit:
                hard_limit_b = human_readable_to_bytes(hard_limit)
            memory_baseline = get_memory_usage()
            no_warning_previous = True
            thread = ThreadWithReturnValue(target=func, args=args, kwargs=kwargs)
            thread.start()
            while True:
                memory = get_memory_usage() - memory_baseline
                if hard_limit and (memory > hard_limit_b):
                    memory_human_r = bytes_to_human_readable(memory)
                    raise MemoryError(f"Occupied memory is {memory_human_r} while hard limit is {hard_limit}")
                elif soft_limit and no_warning_previous and (memory > soft_limit_b):
                    no_warning_previous = False
                    memory_human_r = bytes_to_human_readable(memory)
                    warnings.warn(f"Occupied memory is {memory_human_r} while soft limit is {soft_limit}")                                   
                time.sleep(poll_interval)
                
                if not thread.is_alive():
                    break

            return thread.value

        return inner_func
    return decorator


# In[38]:


@memory_limit(soft_limit="500M", hard_limit="1000M", poll_interval=1)
def memory_increment() -> list:
    """
    Test function. Reaches 1.89G memory occupation within several seconds.

    Args:
        None

    Returns:
        List of numbers.
    """
    lst = []
    for i in range(50000000):
        if i % 500000 == 0:
            time.sleep(0.1)
        lst.append(i)
    return lst


# In[39]:


memory_increment()


# # Задание 3 (11 баллов)

# Напишите функцию `parallel_map`. Это должна быть **универсальная** функция для распараллеливания, которая эффективно работает в любых условиях.
# 
# Функция должна принимать следующие аргументы:
# 1. `target_func` - целевая функция (обязательный аргумент)
# 2. `args_container` - контейнер с позиционными аргументами для `target_func` (по-умолчанию `None` - позиционные аргументы не передаются)
# 3. `kwargs_container` - контейнер с именованными аргументами для `target_func` (по-умолчанию `None` - именованные аргументы не передаются)
# 4. `n_jobs` - количество workers, которые будут использованы для выполнения (по-умолчанию `None` - количество логических ядер CPU в системе)
# 
# Функция должна работать аналогично `***PoolExecutor.map`, применяя функцию к переданному набору аргументов, но с некоторыми дополнениями и улучшениями
#     
# Поскольку мы пишем **универсальную** функцию, то нам нужно будет выполнить ряд требований, чтобы она могла логично и эффективно работать в большинстве ситуаций
# 
# 1. `target_func` может принимать аргументы любого вида в любом количестве
# 2. Любые типы данных в `args_container`, кроме `tuple`, передаются в `target_func` как единственный позиционный аргумент. `tuple` распаковываются в несколько аргументов
# 3. Количество элементов в `args_container` должно совпадать с количеством элементов в `kwargs_container` и наоборот, также значение одного из них или обоих может быть равно `None`, в иных случаях должна кидаться ошибка (оба аргумента переданы, но размеры не совпадают)
# 
# 4. Функция должна выполнять определённое количество параллельных вызовов `target_func`, это количество зависит от числа переданных аргументов и значения `n_jobs`. Сценарии могут быть следующие
#     + `args_container=None`, `kwargs_container=None`, `n_jobs=None`. В таком случае функция `target_func` выполнится параллельно столько раз, сколько на вашем устройстве логических ядер CPU
#     + `args_container=None`, `kwargs_container=None`, `n_jobs=5`. В таком случае функция `target_func` выполнится параллельно **5** раз
#     + `args_container=[1, 2, 3]`, `kwargs_container=None`, `n_jobs=5`. В таком случае функция `target_func` выполнится параллельно **3** раза, несмотря на то, что `n_jobs=5` (так как есть всего 3 набора аргументов для которых нам нужно получить результат, а лишние worker'ы создавать не имеет смысла)
#     + `args_container=None`, `kwargs_container=[{"s": 1}, {"s": 2}, {"s": 3}]`, `n_jobs=5`. Данный случай аналогичен предыдущему, но здесь мы используем именованные аргументы
#     + `args_container=[1, 2, 3]`, `kwargs_container=[{"s": 1}, {"s": 2}, {"s": 3}]`, `n_jobs=5`. Данный случай аналогичен предыдущему, но здесь мы используем и позиционные, и именованные аргументы
#     + `args_container=[1, 2, 3, 4]`, `kwargs_container=None`, `n_jobs=2`. В таком случае в каждый момент времени параллельно будет выполняться **не более 2** функций `target_func`, так как нам нужно выполнить её 4 раза, но у нас есть только 2 worker'а.
#     + В подобных случаях (из примера выше) должно оптимизироваться время выполнения. Если эти 4 вызова выполняются за 5, 1, 2 и 1 секунды, то параллельное выполнение с `n_jobs=2` должно занять **5 секунд** (не 7 и тем более не 10)
# 
# 5. `parallel_map` возвращает результаты выполнения `target_func` **в том же порядке**, в котором были переданы соответствующие аргументы
# 6. Работает с функциями, созданными внутри других функций
# 
# Для базового решения от вас не ожидается **сверххорошая** оптимизация по времени и памяти для всех возможных случаев. Однако за хорошо оптимизированную логику работы можно получить до **+3 дополнительных баллов**
# 
# Вы можете сделать класс вместо функции, если вам удобнее
# 
# В задании можно использовать только модули из **стандартной библиотеки** питона
# 
# Ниже приведены тестовые примеры по каждому из требований

# In[22]:


def parallel_map(target_func,
                 args_container=None,
                 kwargs_container=None,
                 n_jobs=None):
    # Ваш код здесь


# In[156]:


import time


# Это только один пример тестовой функции, ваша parallel_map должна уметь эффективно работать с ЛЮБЫМИ функциями
# Поэтому обязательно протестируйте код на чём-нибудбь ещё
def test_func(x=1, s=2, a=1, b=1, c=1):
    time.sleep(s)
    return a*x**2 + b*x + c


# In[157]:


get_ipython().run_cell_magic('time', '', '\n# Пример 2.1\n# Отдельные значения в args_container передаются в качестве позиционных аргументов\nparallel_map(test_func, args_container=[1, 2.0, 3j-1, 4])   # Здесь происходят параллельные вызовы: test_func(1) test_func(2.0) test_func(3j-1) test_func(4)\n')


# In[158]:


get_ipython().run_cell_magic('time', '', '\n# Пример 2.2\n# Элементы типа tuple в args_container распаковываются в качестве позиционных аргументов\nparallel_map(test_func, [(1, 1), (2.0, 2), (3j-1, 3), 4])    # Здесь происходят параллельные вызовы: test_func(1, 1) test_func(2.0, 2) test_func(3j-1, 3) test_func(4)\n')


# In[159]:


get_ipython().run_cell_magic('time', '', '\n# Пример 3.1\n# Возможна одновременная передача args_container и kwargs_container, но количества элементов в них должны быть равны\nparallel_map(test_func,\n             args_container=[1, 2, 3, 4],\n             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}, {"s": 3}])\n\n# Здесь происходят параллельные вызовы: test_func(1, s=3) test_func(2, s=3) test_func(3, s=3) test_func(4, s=3)\n')


# In[42]:


get_ipython().run_cell_magic('time', '', '\n# Пример 3.2\n# args_container может быть None, а kwargs_container задан явно\nparallel_map(test_func,\n             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}, {"s": 3}])\n')


# In[43]:


get_ipython().run_cell_magic('time', '', '\n# Пример 3.3\n# kwargs_container может быть None, а args_container задан явно\nparallel_map(test_func,\n             args_container=[1, 2, 3, 4])\n')


# In[44]:


get_ipython().run_cell_magic('time', '', '\n# Пример 3.4\n# И kwargs_container, и args_container могут быть не заданы\nparallel_map(test_func)\n')


# In[44]:


get_ipython().run_cell_magic('time', '', '\n# Пример 3.4\n# И kwargs_container, и args_container могут быть не заданы\nparallel_map(test_func)\n')


# In[32]:


get_ipython().run_cell_magic('time', '', '\n# Пример 3.5\n# При несовпадении количеств позиционных и именованных аргументов кидается ошибка\nparallel_map(test_func,\n             args_container=[1, 2, 3, 4],\n             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}])\n')


# In[45]:


get_ipython().run_cell_magic('time', '', '\n# Пример 4.1\n# Если функция не имеет обязательных аргументов и аргумент n_jobs не был передан, то она выполняется параллельно столько раз, сколько ваш CPU имеет логических ядер\n# В моём случае это 24, у вас может быть больше или меньше\nparallel_map(test_func)\n')


# In[47]:


get_ipython().run_cell_magic('time', '', '\n# Пример 4.2\n# Если функция не имеет обязательных аргументов и передан только аргумент n_jobs, то она выполняется параллельно n_jobs раз\nparallel_map(test_func, n_jobs=2)\n')


# In[48]:


get_ipython().run_cell_magic('time', '', "\n# Пример 4.3\n# Если аргументов для target_func указано МЕНЬШЕ, чем n_jobs, то используется такое же количество worker'ов, сколько было передано аргументов\nparallel_map(test_func,\n             args_container=[1, 2, 3],\n             n_jobs=5)   # Здесь используется 3 worker'a\n")


# In[49]:


get_ipython().run_cell_magic('time', '', '\n# Пример 4.4\n# Аналогичный предыдущему случай, но с именованными аргументами\nparallel_map(test_func,\n             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}],\n             n_jobs=5)   # Здесь используется 3 worker\'a\n')


# In[50]:


get_ipython().run_cell_magic('time', '', '\n# Пример 4.5\n# Комбинация примеров 4.3 и 4.4 (переданы и позиционные и именованные аргументы)\nparallel_map(test_func,\n             args_container=[1, 2, 3],\n             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}],\n             n_jobs=5)   # Здесь используется 3 worker\'a\n')


# In[50]:


get_ipython().run_cell_magic('time', '', "\n# Пример 4.6\n# Если аргументов для target_func указано БОЛЬШЕ, чем n_jobs, то используется n_jobs worker'ов\nparallel_map(test_func,\n             args_container=[1, 2, 3, 4],\n             kwargs_container=None,\n             n_jobs=2)   # Здесь используется 2 worker'a\n")


# In[51]:


get_ipython().run_cell_magic('time', '', '\n# Пример 4.7\n# Время выполнения оптимизируется, данный код должен отрабатывать за 5 секунд\nparallel_map(test_func,\n             kwargs_container=[{"s": 5}, {"s": 1}, {"s": 2}, {"s": 1}],\n             n_jobs=2)\n')


# In[57]:


def test_func2(string, sleep_time=1):
    time.sleep(sleep_time)
    return string

# Пример 5
# Результаты возвращаются в том же порядке, в котором были переданы соответствующие аргументы вне зависимости от того, когда завершился worker
arguments = ["first", "second", "third", "fourth", "fifth"]
parallel_map(test_func2,
             args_container=arguments,
             kwargs_container=[{"sleep_time": 5}, {"sleep_time": 4}, {"sleep_time": 3}, {"sleep_time": 2}, {"sleep_time": 1}])


# In[58]:


get_ipython().run_cell_magic('time', '', '\n\ndef test_func3():\n    def inner_test_func(sleep_time):\n        time.sleep(sleep_time)\n    return parallel_map(inner_test_func, args_container=[1, 2, 3])\n\n# Пример 6\n# Работает с функциями, созданными внутри других функций\ntest_func3()\n')

