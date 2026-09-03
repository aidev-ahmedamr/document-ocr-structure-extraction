import re


DATE_PATTERNS = [
    r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
    r'\b(\d{4}-\d{2}-\d{2})\b',
    r'\b([A-Z][a-z]{2,8}\s+\d{1,2},?\s+\d{4})\b',
]

# Only treat a number as a monetary amount if it's tied to a currency
# symbol/code or a "Total/Subtotal/Amount" label - otherwise plain
# digit sequences (phone numbers, invoice numbers) get misread as amounts.
AMOUNT_PATTERN = (
    r'(?:(?:USD|EGP|\$|€|£)\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d{4,}(?:\.\d{2})?))'
    r'|'
    r'(?:(?:Total|Subtotal|Amount|Sum)\s*:?\s*(?:USD|EGP|\$|€|£)?\s*'
    r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d{4,}(?:\.\d{2})?))'
)

EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

PHONE_PATTERN = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'

ID_PATTERN = r'\b(?:INV|ID|NO|REF)\b[-#:\s]+([A-Z0-9-]{4,15})'


def extract_dates(text):
    dates = []
    for pattern in DATE_PATTERNS:
        dates.extend(re.findall(pattern, text))
    return list(dict.fromkeys(dates))


def extract_amounts(text):
    matches = re.findall(AMOUNT_PATTERN, text, re.IGNORECASE)
    amounts = []
    for group1, group2 in matches:
        raw = group1 or group2
        if not raw:
            continue
        try:
            amounts.append(float(raw.replace(',', '')))
        except ValueError:
            continue
    return sorted(set(amounts), reverse=True)


def extract_emails(text):
    return list(dict.fromkeys(re.findall(EMAIL_PATTERN, text)))


def extract_phones(text):
    return list(dict.fromkeys(re.findall(PHONE_PATTERN, text)))


def extract_document_ids(text):
    return list(dict.fromkeys(re.findall(ID_PATTERN, text, re.IGNORECASE)))


def extract_fields(text):
    """Run all regex extractors and return a unified fields dict."""
    return {
        "dates": extract_dates(text),
        "amounts": extract_amounts(text),
        "emails": extract_emails(text),
        "phones": extract_phones(text),
        "document_ids": extract_document_ids(text),
    }
