import os
import codecs
import logging
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from readability.readability import Unparseable
from readability.readability import Document as Paper

CAT_PATTERN = r'([a-z_\s]+)/.*'
DOC_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.json'
TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p', 'li']

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

class HTMLCorpusReader(CategorizedCorpusReader, CorpusReader):

    """
    Объект чтения корпуса с HTML-документами для получения
    возможности дополнительной предварительной обработки.
    """
    def __init__(self, root, fileids = DOC_PATTERN, encoding = 'utf8',
                 tags = TAGS, **kwargs):
        """
        Инициализирует объект чтения корпуса.
        Аргументы, управляющие классификацией
        (``cat_pattern``, ``cat_map`` и ``cat_file``), передаются
        в конструктор ``CategorizedCorpusReader``. остальные аргументы
        передаются в конструктор ``CorpusReader``.
        """
        # Добавить шаблон категорий, если он не был передан в класс явно.
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN
        # Инициализировать объекты чтения корпуса из NLTK
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids, encoding)
        # Сохранить теги, подлежащие извлечению.
        self.tags = tags

    def resolve(self, fileids, categories):
        """
        Возвращает список идентификаторов файлов или названий категорий,
        которые передаются каждой внутренней функции объекта чтения корпуса.
        Реализована по аналогии с ``CategorizedPlaintextCorpusReader`` в NLTK.
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")
        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids = None, categories = None):
        """
        Возвращает полный текст HTML-документа, закрывая его
        по завершении чтения.
        """
        # Получить список файлов для чтения
        fileids = self.resolve(fileids, categories)
        # Создать генератор, загружающий документы в память по одному.
        for path, encoding in self.abspaths(fileids, include_encoding = True):
            with codecs.open(path, 'r', encoding = encoding) as f:
                yield f.read()

    def sizes(self, fileids = None, categories = None):
        """
        Возвращает список кортежей, идентификатор файла и его размер.
        Эта функция используется для выявления необычно больших файлов
        в корпусе.
        """
        # Получить список файлов
        fileids = self.resolve(fileids, categories)
        # Создать генератор, возвращающий имена и размеры файлов
        for path in self.abspaths(fileids):
            yield path, os.path.getsize(path)

    def html(self, fileids = None, categories = None):
        """
        Возвращает содержимое HTML каждого документа, очищая его
        с помощью библиотеки readability-lxml.
        """
        for doc in self.docs(fileids, categories):
            try:
                yield Paper(doc).summary()
            except Unparseable as e:
                print("Could not parse HTML: {}".format(e))
                continue
