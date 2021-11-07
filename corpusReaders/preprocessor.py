import os
import pickle
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize


class Preprocessor(object):
    """
    Обертывает `HTMLCorpusReader` и выполняет лексемизацию
    с маркировкой частями речи.
    """
    def __init__(self, corpus, target=None, **kwargs):
        self.corpus = corpus
        self.target = target

    def fileids(self, fileids=None, categories=None):
        fileids = self.corpus.resolve(fileids, categories)
        if fileids:
            return fileids
        return self.corpus.fileids()

    def abspath(self, fileid):
        # Найти путь к каталогу относительно корня исходного корпуса.
        parent = os.path.relpath(
            os.path.dirname(self.corpus.abspath(fileid)), self.corpus.root
        )
        # Выделить части пути для реконструирования
        basename = os.path.basename(fileid)
        name, ext = os.path.splitext(basename)
        # Сконструировать имя файла с расширением .pickle
        basename = name + '.pickle'
        # Вернуть путь к файлу относительно корня целевого корпуса.
        return os.path.normpath(os.path.join(self.target, parent, basename))

    def tokenize(self, fileid):
        for paragraph in self.corpus.paras(fileids=fileid):
            yield [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(paragraph)
            ]

    def process(self, fileid):
        """
        Вызывается для одного файла, проверяет местоположение на диске,
        чтобы гарантировать отсутствие ошибок, использует +tokenize()+ для
        предварительной обработки и записывает трансформированный документ
        в виде сжатого архива в заданное место.
        """
        # Определить путь к файлу для записи результата.
        target = self.abspath(fileid)
        parent = os.path.dirname(target)
        # Убедиться в существовании каталога
        if not os.path.exists(parent):
            os.makedirs(parent)
        # Убедиться, что parent — это каталог, а не файл
        if not os.path.isdir(parent):
            raise ValueError(
                "Please supply a directory to write preprocessed data to."
            )
        # Создать структуру данных для записи в архив
        document = list(self.tokenize(fileid))
        # Записать данные в архив на диск
        with open(target, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)
        # Удалить документ из памяти
        del document
        # Вернуть путь к целевому файлу
        return target

    def transform(self, fileids=None, categories=None):
        # Создать целевой каталог, если его еще нет
        if not os.path.exists(self.target):
            os.makedirs(self.target)
        # Получить имена файлов для обработки
        for fileid in self.fileids(fileids, categories):
            yield self.process(fileid)
