
import re
import unicodedata


# Arabic-to-Persian character mapping
CHAR_MAP = str.maketrans({
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
    "ۀ": "ه",
    "ة": "ه",
    "ؤ": "و",
    "إ": "ا",
    "أ": "ا",
    "ٱ": "ا",
})


# Convert Persian and Arabic digits to English digits
DIGIT_MAP = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789"
)


# Remove Arabic diacritics
DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

# Remove invisible formatting characters
INVISIBLE_RE = re.compile(
    r"[\u200B\u200D-\u200F\u202A-\u202E\u2060\uFEFF]"
)

TATWEEL = "\u0640"


# Reduce three or more repeated letters to one
REPEATED_LETTERS_RE = re.compile(
    r"([^\W\d_])\1{2,}",
    flags=re.UNICODE
)


# Keep letters, digits, whitespace, and half-space
PUNCTUATION_RE = re.compile(
    r"[^\w\s\u200c]|_",
    flags=re.UNICODE
)


# Normalize whitespace around half-spaces
HALF_SPACE_RE = re.compile(r"\s*\u200c\s*")

# Collapse consecutive whitespace
WHITESPACE_RE = re.compile(r"\s+")


def normalize_persian_text(text: str) -> str:
    """
    Normalize Persian text for NLP preprocessing.
    """

    if text is None:
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Standardize Arabic/Persian letters and digits
    text = text.translate(CHAR_MAP)
    text = text.translate(DIGIT_MAP)

    # Remove diacritics, tatweel, and invisible characters
    text = DIACRITICS_RE.sub("", text)
    text = text.replace(TATWEEL, "")
    text = INVISIBLE_RE.sub("", text)

    # Reduce elongated words
    text = REPEATED_LETTERS_RE.sub(r"\1", text)

    # Replace punctuation with spaces
    text = PUNCTUATION_RE.sub(" ", text)

    # Normalize half-spaces
    text = HALF_SPACE_RE.sub("\u200c", text)

    # Normalize whitespace
    text = WHITESPACE_RE.sub(" ", text).strip()

    return text