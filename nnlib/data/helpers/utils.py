from typing import List
import inflect
from unidecode import unidecode
import re

__all__ = ["tokenize_line_by_word",
           "english_cleaners",
           "basic_cleaners",
           "transliteration_cleaners",
           "convert_to_ascii",
           "collapse_whitespace",
           "normalize_numbers",
           "expand_abbreviations",
           "remove_punctuation"
           ]


def tokenize_line_by_word(line: str, lowercase: bool = False) -> List[str]:
    line = re.compile(r"\s+").sub(" ", line)  # replace space-like characters to " "
    line = line.strip()  # remove front/end trailing spaces
    if lowercase:
        line = line.lower()
    return line.split()  # split by space


# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper/common/text/numbers.py
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper/common/text/cleaners.py

_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')
_time_re = re.compile(r'([0-9]{1,2}):([0-9]{2})')
_whitespace_re = re.compile(r'\s+')

_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]

# do we need this?
_corrections = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ("sharp'st", "sharpest"),
    ("need'st", "needest"),
    ("evenin's", "evenings"),
    ("feelin's", "feelings"),
    ("ev'ybody", "everybody"),
    ("fam'ly", "family"),
    ("remov'd", "removed"),
    ("pierc'd", "pierced"),
    ("steel'd", "steeled"),
    ("poison'd", "poisoned"),
    ("millon'd", "millioned"),
    ("rebuk'd", "rebuked"),
    ("emerg'd", "emerged"),
    ("link'd", "linked"),
    ("impress'd", "impressed"),
    ("resign'd", "resigned"),
    ("look'd", "looked"),
    ("possess'd", "possessed"),
    ("stepp'd", "stepped"),
    ("laugh'd", "laughed"),
    ("chopp'd", "chopped"),
    ("cross'd", "crossed"),
    ("seem'd", "seemed"),
    ("could n't", "couldn't"),
    ("had n't", "hadn't"),
    ("has n't", "hasn't"),
]]


def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    if int(m.group(0)[0]) == 0:
        return _inflect.number_to_words(m.group(0), andword='', group=1)
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        elif num > 2000 and num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    # Add check for number phones and other large numbers
    elif num > 1000000000 and num % 10000 != 0:
        return _inflect.number_to_words(num, andword='', group=1)
    else:
        return _inflect.number_to_words(num, andword='')


def _expand_time(m):
    mins = int(m.group(2))
    if mins == 0:
        return _inflect.number_to_words(m.group(1))
    return " ".join([_inflect.number_to_words(m.group(1)), _inflect.number_to_words(m.group(2))])


def normalize_numbers(text: str) -> str:
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    text = re.sub(_time_re, _expand_time, text)
    return text


def expand_abbreviations(text: str) -> str:
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    # for regex, replacement in _corrections:
    #     text = re.sub(regex, replacement, text)
    return text


def collapse_whitespace(text: str) -> str:
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text: str) -> str:
    return unidecode(text)


def remove_punctuation(text: str, table) -> str:
    text = text.translate(table)
    text = re.sub(r'&', " and ", text)
    text = re.sub(r'\+', " plus ", text)
    return text


def basic_cleaners(text: str, lowercase: bool = False) -> str:
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = collapse_whitespace(text)
    text = text.lower() if lowercase else text.upper()
    return text


def transliteration_cleaners(text: str, lowercase: bool = False) -> str:
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = collapse_whitespace(text)
    text = text.lower() if lowercase else text.upper()
    return text


def english_cleaners(text, table=None, lowercase: bool = False) -> str:
    """Pipeline for English text, including number and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    if table is not None:
        text = remove_punctuation(text, table)
    text = collapse_whitespace(text)
    text = text.lower() if lowercase else text.upper()
    return text
