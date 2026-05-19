# Arabic text preprocessing helpers.
# Used by the Phase 2 preprocessing notebook and the streaming pipeline.
# Note: when used inside Spark UDFs, the equivalent logic is defined inline
# in each notebook so Spark can serialize the function to its workers.

import re
import pyarabic.araby as araby


# Arabic stopwords commonly used in academic abstracts.
# Includes prepositions, conjunctions, pronouns, demonstratives, and modal words.
ARABIC_STOPWORDS = {
    "في", "من", "إلى", "على", "عن", "حتى", "منذ", "مذ", "خلال",
    "بين", "أمام", "خلف", "تحت", "فوق",
    "و", "ف", "ثم", "أو", "أم", "بل", "لكن", "كما", "حيث",
    "إذا", "إذ", "إن", "أن", "كان",
    "هو", "هي", "هم", "هن", "أنا", "نحن", "أنت", "أنتم",
    "ذلك", "تلك", "هذا", "هذه", "هؤلاء",
    "ها", "هما", "أنتما", "إياك", "إياه", "إياها",
    "كذلك", "أيضا", "بعض", "كل", "جميع", "بعد", "قبل",
    "حول", "نحو", "ضد",
    "قد", "لقد", "لا", "ما", "لم", "لن", "ليس", "غير", "سوى",
    "كانت", "يكون", "تكون", "أصبح", "صار", "ظل", "بات",
    "إذن", "لذلك", "بحيث", "لذا",
    "أولا", "ثانيا", "ثالثا", "رابعا", "خامسا",
}


def remove_diacritics(text: str) -> str:
    # Strip tashkeel (fatha, kasra, damma, sukun, shadda, tanwin variants).
    if not text:
        return ""
    return araby.strip_tashkeel(text)


def normalize_arabic(text: str) -> str:
    # Unify alef variants, yeh variants, and remove tatweel.
    if not text:
        return ""
    text = araby.strip_tatweel(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    return text


def remove_non_arabic(text: str) -> str:
    # Keep only Arabic characters (U+0600 to U+06FF) and whitespace.
    if not text:
        return ""
    text = re.sub(r"[^؀-ۿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    if not text:
        return ""
    tokens = text.split()
    filtered = [w for w in tokens if w not in ARABIC_STOPWORDS]
    return " ".join(filtered)


def preprocess_text(text: str) -> str:
    # Full preprocessing pipeline:
    # 1. Remove diacritics
    # 2. Normalize Arabic characters
    # 3. Remove non-Arabic characters (Latin letters, digits, punctuation)
    # 4. Remove stopwords
    if text is None:
        return ""
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = remove_non_arabic(text)
    text = remove_stopwords(text)
    return text


if __name__ == "__main__":
    sample = "هذا مِثَالٌ عَلَى نَصٍّ عَرَبِيٍّ يَحْتَوِي عَلَى تَشْكِيلٍ وَأَرْقَامٍ 123 وَكَلِمَاتٍ English."
    print("Original:    ", sample)
    print("Diacritics:  ", remove_diacritics(sample))
    print("Normalized:  ", normalize_arabic(remove_diacritics(sample)))
    print("Arabic only: ", remove_non_arabic(normalize_arabic(remove_diacritics(sample))))
    print("Full clean:  ", preprocess_text(sample))
