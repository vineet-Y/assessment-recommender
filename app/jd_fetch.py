import re
import io
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text

HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_text_from_url(url):

    r = requests.get(url, headers=HEADERS, timeout=20)

    r.raise_for_status()

    if ".pdf" in url:

        text = extract_text(io.BytesIO(r.content))

        return clean(text)

    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(" ")

    return clean(text)


def clean(t):

    t = re.sub(r"\s+", " ", t)

    return t[:100000]