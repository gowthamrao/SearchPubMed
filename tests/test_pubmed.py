import pytest
import pandas as pd
import requests
from searchpubmed import pubmed

class DummyResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code
        self.ok = (status_code == 200)
        self.text = content.decode() if isinstance(content, bytes) else content

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(response=self)


def test_get_pmid_from_pubmed_empty(monkeypatch):
    # Simulate network error to trigger empty return
    monkeypatch.setattr('searchpubmed.pubmed.requests.post', 
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            requests.RequestException("Network error")))
    result = pubmed.get_pmid_from_pubmed("test query")
    assert result == []


def test_get_pmid_from_pubmed_parse(monkeypatch):
    # Simulate successful API response with duplicate IDs
    xml = b"""<?xml version='1.0'?><eSearchResult><IdList>
        <Id>123</Id><Id>456</Id><Id>123</Id>
    </IdList></eSearchResult>"""
    resp = DummyResponse(xml)
    monkeypatch.setattr('searchpubmed.pubmed.requests.post', lambda *args, **kwargs: resp)
    result = pubmed.get_pmid_from_pubmed("query")
    assert result == ["123", "456"]


def test_map_pmids_to_pmcids_empty():
    df = pubmed.map_pmids_to_pmcids([])
    assert list(df.columns) == ["pmid", "pmcid"]
    assert df.empty


def test_get_pmc_full_xml_empty():
    df = pubmed.get_pmc_full_xml([])
    assert list(df.columns) == ["pmcid", "fullXML"]
    assert df.empty


def test_get_pmc_html_text_empty():
    df = pubmed.get_pmc_html_text([])
    assert list(df.columns) == ["pmcid", "htmlText", "scrapeMsg"]
    assert df.empty


def test_get_pmc_full_text_fallback(monkeypatch):
    # Simulate flat HTML too short, then XML fallback
    def fake_get(url, headers=None, timeout=None):
        class R:
            def __init__(self, text=None, content=None):
                self.text = text
                self.content = content
                self.status_code = 200
            def raise_for_status(self):
                pass
        if '?format=flat' in url:
            return R(text='<p>Short</p>')
        # XML fallback
        xml_body = b'<article><body><p>Full text here</p></body></article>'
        return R(content=xml_body)

    monkeypatch.setattr('searchpubmed.pubmed.requests.get', fake_get)
    # With xml_fallback_min_chars > len(flat) to force fallback
    out = pubmed.get_pmc_full_text("1234", xml_fallback_min_chars=10)
    assert out == {"PMC1234": "Full text here"}


def test_fetch_pubmed_fulltexts_empty(monkeypatch):
    monkeypatch.setattr('searchpubmed.pubmed.get_pmid_from_pubmed', 
                        lambda query, **kwargs: [])
    df = pubmed.fetch_pubmed_fulltexts("anything")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
