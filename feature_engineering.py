# Функции извлечения признаков - те же самые, что использовались при обучении модели в ноутбуке (шаги 6-8), чтобы новый текст обрабатывался точно так же, как обучающие данные.
import re
import string
from wordcloud import STOPWORDS

# тот же набор стоп-слов, что использовался при обучении
STOP_WORDS = set(STOPWORDS)

ENGINEERED_COLS = [
    'char_count', 'word_count', 'sentence_count', 'avg_word_length',
    'unique_words', 'lexical_diversity', 'stopword_count',
    'punctuation_count', 'uppercase_count', 'digit_count',
]

ENGINEERED_COLS_DESCRIPTIONS = {
    'char_count': 'Количество символов в тексте',
    'word_count': 'Количество слов',
    'sentence_count': 'Количество предложений (по точкам/!/?)',
    'avg_word_length': 'Средняя длина слова',
    'unique_words': 'Количество уникальных слов',
    'lexical_diversity': 'Доля уникальных слов от общего числа слов',
    'stopword_count': 'Количество стоп-слов (the, a, is и т.п.)',
    'punctuation_count': 'Количество знаков препинания',
    'uppercase_count': 'Количество заглавных букв',
    'digit_count': 'Количество цифр',
}


def clean_text(text):
    # удаление служебных символов переноса строки \r\n
    text = re.sub(r'\r\n', ' ', text)
    return text.strip()


def extract_features(text):
    # считает те же 10 инженерных признаков, что и в ноутбуке
    words = text.split()
    word_tokens = re.findall(r"[a-zA-Z']+", text.lower())

    avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
    lexical_diversity = len(set(word_tokens)) / len(word_tokens) if word_tokens else 0

    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len(re.findall(r'[.!?]+', text)),
        'avg_word_length': avg_word_length,
        'unique_words': len(set(word_tokens)),
        'lexical_diversity': lexical_diversity,
        'stopword_count': sum(1 for w in word_tokens if w in STOP_WORDS),
        'punctuation_count': sum(1 for ch in text if ch in string.punctuation),
        'uppercase_count': sum(1 for ch in text if ch.isupper()),
        'digit_count': sum(1 for ch in text if ch.isdigit()),
    }