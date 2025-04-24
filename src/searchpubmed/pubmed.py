from __future__ import annotations

##############################################################################
#  Imports & logger                                                          #
##############################################################################
import logging
import re
import time
import xml.etree.ElementTree as ET
from math import ceil
from typing import List, Dict

import dateparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.exceptions import HTTPError, RequestException

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_pmid_from_pubmed(
    query: str,
    *,
    retmax: int = 2_000,
    api_key: str | None = None,
    timeout: int = 20,
    max_retries: int = 5,
    delay: float = 0.34,
) -> list[str]:
    """
    ------------------------------------------------------------------------
    Return a list of PubMed IDs (PMIDs) for an arbitrary search expression.
    ------------------------------------------------------------------------

    Parameters
    ----------
    query : str
        Any valid PubMed search term (Boolean logic, field tags, etc.).
    retmax : int, default 2 000
        Maximum number of PMIDs to retrieve (ESearch hard-cap = 100 000).
    api_key : str | None, optional
        NCBI API key – raises the personal rate limit to ~10 req s⁻¹.
    timeout : int, default 20 s
        Socket timeout for the HTTP request.
    max_retries : int, default 5
        How many times to retry on HTTP 429 or 5xx errors.
    delay : float, default 0.34 s
        Base pause between successive retries (doubles each attempt).

    Returns
    -------
    list[str]
        Unique PMIDs (as strings).  Empty list if the query matches none or
        if all retries fail.

    Notes
    -----
    *   The function is deliberately lightweight – no pandas dependency.
    *   It logs its progress and failures through the *logging* module; hook
        this into your existing logger configuration if needed.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(base_url, data=params, timeout=timeout)
            resp.raise_for_status()
            break  # success
        except HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status and (status == 429 or 500 <= status < 600) and attempt < max_retries:
                wait = delay * (2 ** (attempt - 1))
                logger.warning(
                    f"ESearch HTTP {status}; retry {attempt}/{max_retries} in {wait:.1f}s"
                )
                time.sleep(wait)
                continue
            logger.error(f"ESearch failed (HTTP {status}): {e}")
            return []
        except RequestException as e:
            logger.error(f"ESearch network error: {e}")
            return []

    try:
        root = ET.fromstring(resp.content)
        pmids = [id_el.text for id_el in root.findall(".//IdList/Id") if id_el.text]
        # Deduplicate while preserving order
        seen: set[str] = set()
        return [p for p in pmids if not (p in seen or seen.add(p))]
    except ET.ParseError as e:
        logger.error(f"ESearch XML parse error: {e}")
        return []




def map_pmids_to_pmcids(
    pmids: List[str],
    *,
    api_key: str | None = None,
    batch_size: int = 500,
    timeout: int = 20,
    max_retries: int = 5,
    delay: float = 0.34,
) -> pd.DataFrame:
    """
    ------------------------------------------------------------------------
    Map **PubMed IDs (PMIDs)** to **all** corresponding **PMC IDs (PMCIDs)**.
    ------------------------------------------------------------------------

    Parameters
    ----------
    pmids : list[str]
        PMIDs to map.  Duplicates are tolerated and preserved in the output.
    api_key : str | None, optional
        NCBI API key (raises personal limit to ≈10 req s⁻¹).
    batch_size : int, default 500
        Number of PMIDs sent per ELink request (hard cap = 2 000).
    timeout : int, default 20 s
        Socket timeout per HTTP request.
    max_retries : int, default 5
        Attempts per batch on HTTP 429 / 5xx before falling back to
        `pmcid = <NA>` for the affected PMIDs.
    delay : float, default 0.34 s
        Base pause between retries (exponential back-off).

    Returns
    -------
    pandas.DataFrame
        Columns
        --------
        ``pmid``   | string  
        ``pmcid``  | string  (``<NA>`` if no PMC record exists)

        The frame has as many rows as there are *actual* mappings; e.g.  
        *one PMID with three PMCIDs* → *three rows*.

    Notes
    -----
    * Uses **ELink** (`dbfrom=pubmed`, `db=pmc`) under the hood.
    * Rate-limit friendly (≤3 req s⁻¹ without key, ~10 req s⁻¹ with key).
    * XML parse errors on individual batches degrade gracefully to
      ``pmcid = <NA>`` for the PMIDs in that batch.
    """
    # ── Guard clause ────────────────────────────────────────────
    if not pmids:
        return pd.DataFrame(columns=["pmid", "pmcid"]).astype("string")

    base_elink = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    session = requests.Session()
    records: list[tuple[str, str | None]] = []

    total_batches = ceil(len(pmids) / batch_size)

    for idx in range(total_batches):
        chunk = pmids[idx * batch_size : (idx + 1) * batch_size]

        # Build URL-encoded body once; add each PMID as a separate "id"
        data = [
            ("dbfrom", "pubmed"),
            ("db", "pmc"),
            ("retmode", "xml"),
        ]
        if api_key:
            data.append(("api_key", api_key))
        data.extend(("id", pmid) for pmid in chunk)

        logger.info(f"ELink batch {idx+1}/{total_batches} (size={len(chunk)})")

        # ── HTTP with retry ─────────────────────────────────────
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = session.post(base_elink, data=data, timeout=timeout)
                if response.status_code == 429:
                    raise HTTPError(response=response)
                response.raise_for_status()
                break                                           # success
            except (HTTPError, RequestException) as exc:
                status = getattr(exc.response, "status_code", None)
                if status and (status == 429 or 500 <= status < 600) and attempt < max_retries:
                    wait = delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"Batch {idx+1}: HTTP {status}, retry {attempt}/{max_retries} in {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"Batch {idx+1} failed: {exc}")
                break                                           # fall through

        # On total failure → all PMIDs in chunk get <NA>
        if response is None or not response.ok:
            records.extend((pmid, None) for pmid in chunk)
            continue

        # ── XML parse ──────────────────────────────────────────
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as e:
            logger.error(f"XML parse error for batch {idx+1}: {e}")
            records.extend((pmid, None) for pmid in chunk)
            time.sleep(delay)
            continue

        # ── Extract mappings ──────────────────────────────────
        for linkset in root.findall("LinkSet"):
            pmid_text = linkset.findtext("IdList/Id")
            if not pmid_text:
                continue
            pmcids = [
                link.text
                for db in linkset.findall("LinkSetDb")
                if db.findtext("DbTo") == "pmc"
                for link in db.findall("Link/Id")
                if link.text
            ]
            if pmcids:
                records.extend((pmid_text, pmcid) for pmcid in pmcids)
            else:  # preserve the PMID even if it lacks a PMC record
                records.append((pmid_text, None))

        time.sleep(delay)

    df = pd.DataFrame(records, columns=["pmid", "pmcid"]).astype("string")
    return df



def get_pubmed_metadata(
    pmids: List[str],
    *,
    api_key: str | None = None,
    batch_size: int = 200,
    timeout: int = 20,
    max_retries: int = 3,
    delay: float = 0.34,
) -> pd.DataFrame:
    """
    ------------------------------------------------------------------------
    Fetch structured PubMed metadata for an arbitrary list of PMIDs.
    ------------------------------------------------------------------------

    Parameters
    ----------
    pmids : list[str]
        PubMed IDs to retrieve.  Duplicates are tolerated and de-duplicated.
    api_key : str | None, optional
        NCBI API key (optional but recommended – lifts rate-limit to ≈10 req s⁻¹).
    batch_size : int, default 200
        PMIDs per EFetch request (ceiling = 10 000).
    timeout : int, default 20 s
        Socket timeout per HTTP call.
    max_retries : int, default 3
        Attempts per batch on HTTP-429 / 5xx before giving up.
    delay : float, default 0.34 s
        Base back-off (doubles each retry).

    Returns
    -------
    pandas.DataFrame
        Column               | dtype  | description
        ---------------------|--------|----------------------------------------
        pmid                 | string | PubMed identifier
        title                | string | Article title (sentence case)
        abstract             | string | Abstract (paragraphs joined)
        journal              | string | Full journal title
        publicationDate      | string | ISO-8601 date (YYYY-MM-DD / YYYY-MM)
        doi                  | string | Digital Object Identifier
        firstAuthor          | string | “Given Surname” of first author
        lastAuthor           | string | “Given Surname” of last author
        authorAffiliations   | string | “; ”-separated affiliations
        meshTags             | string | “, ”-separated MeSH descriptors
        keywords             | string | “, ”-separated author keywords

        Missing data are filled with the literal string ``"N/A"`` for
        type-stability.

    Notes
    -----
    * Only **one HTTP round-trip per *batch***; results are concatenated.
    * Parsing failures for individual articles degrade gracefully to rows
      filled with ``"N/A"``.
    """
    # ── Guard clause ────────────────────────────────────────────
    unique_pmids = list(dict.fromkeys(pmids))        # de-dup, keep order
    if not unique_pmids:
        return pd.DataFrame(columns=[
            "pmid", "title", "abstract", "journal", "publicationDate",
            "doi", "firstAuthor", "lastAuthor",
            "authorAffiliations", "meshTags", "keywords",
        ]).astype("string")

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    session = requests.Session()
    records: list[dict] = []

    # ── Helpers ─────────────────────────────────────────────────
    def _parse_pubdate(elem: ET.Element | None) -> str:
        if elem is None:
            return "N/A"
        y = elem.findtext("Year")
        m = elem.findtext("Month") or ""
        d = elem.findtext("Day") or ""
        if y and m:
            try:
                return dateparser.parse(f"{y} {m} {d or '1'}").date().isoformat()
            except Exception:
                return "-".join(p for p in (y, m, d) if p)
        return elem.findtext("MedlineDate") or y or "N/A"

    def _fullname(author: ET.Element) -> str:
        fore = author.findtext("ForeName") or author.findtext("Initials") or ""
        last = author.findtext("LastName") or ""
        name = f"{fore} {last}".strip()
        return name or "N/A"

    # ── Main loop ───────────────────────────────────────────────
    for start in range(0, len(unique_pmids), batch_size):
        batch = unique_pmids[start:start + batch_size]
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "id": ",".join(batch),
        }
        if api_key:
            params["api_key"] = api_key

        # ---- HTTP with retry ----------------------------------
        resp = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = session.get(base_url, params=params, timeout=timeout)
                resp.raise_for_status()
                break
            except HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status and (status == 429 or 500 <= status < 600) and attempt < max_retries:
                    wait = delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"Batch {start//batch_size+1}: HTTP {status}; retry in {wait:.1f}s "
                        f"(attempt {attempt}/{max_retries})"
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"EFetch HTTP error for PMIDs {batch}: {e}")
                break
            except RequestException as e:
                logger.error(f"Network error for PMIDs {batch}: {e}")
                break

        if resp is None or not resp.ok:
            # total failure → placeholder rows
            for pmid in batch:
                records.append({k: "N/A" for k in (
                    "title", "abstract", "journal", "publicationDate", "doi",
                    "firstAuthor", "lastAuthor", "authorAffiliations",
                    "meshTags", "keywords"
                )} | {"pmid": pmid})
            continue

        # ---- XML parse ----------------------------------------
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as e:
            logger.error(f"XML parse error for PMIDs {batch}: {e}")
            for pmid in batch:
                records.append({k: "N/A" for k in (
                    "title", "abstract", "journal", "publicationDate", "doi",
                    "firstAuthor", "lastAuthor", "authorAffiliations",
                    "meshTags", "keywords"
                )} | {"pmid": pmid})
            time.sleep(delay)
            continue

        # ---- Extract article info -----------------------------
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID", default="N/A")

            title = art.findtext(".//ArticleTitle", default="N/A").strip()

            abstract = " ".join(
                t.text or "" for t in art.findall(".//Abstract/AbstractText")
            ).strip() or "N/A"

            journal = art.findtext(".//Journal/Title", default="N/A")

            pubdate_elem = art.find(".//JournalIssue/PubDate")
            publication_date = _parse_pubdate(pubdate_elem)

            doi = art.findtext('.//ArticleIdList/ArticleId[@IdType="doi"]', default="N/A")

            authors = art.findall(".//AuthorList/Author")
            first_author = _fullname(authors[0]) if authors else "N/A"
            last_author = _fullname(authors[-1]) if authors else "N/A"

            affiliations = [
                aff.text for a in authors
                for aff in a.findall("AffiliationInfo/Affiliation") if aff.text
            ]
            author_affiliations = "; ".join(affiliations) or "N/A"

            mesh_tags = ", ".join(
                mh.text for mh in art.findall(".//MeshHeading/DescriptorName") if mh.text
            ) or "N/A"

            keywords = ", ".join(
                kw.text for kw in art.findall(".//KeywordList/Keyword") if kw.text
            ) or "N/A"

            records.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "publicationDate": publication_date,
                "doi": doi,
                "firstAuthor": first_author,
                "lastAuthor": last_author,
                "authorAffiliations": author_affiliations,
                "meshTags": mesh_tags,
                "keywords": keywords,
            })

        time.sleep(delay)

    return (
        pd.DataFrame(records)
          .astype("string")
          .sort_values("pmid", ignore_index=True)
    )




def _strip_default_ns(xml_bytes: bytes) -> bytes:
    """
    Remove the *first* default namespace declaration (xmlns="…") so that the
    returned XML is easy to address with bare tag names.
    """
    return re.sub(rb'\sxmlns="[^"]+"', b"", xml_bytes, count=1)


def get_pmc_full_xml(
    pmcids: List[str],
    *,
    api_key: str | None = None,
    batch_size: int = 200,
    timeout: int = 20,
    max_retries: int = 3,
    delay: float = 0.34,
) -> pd.DataFrame:
    """
    ------------------------------------------------------------------------
    Retrieve the full-text **JATS XML** for one or many PubMed Central IDs.
    ------------------------------------------------------------------------

    Parameters
    ----------
    pmcids : list[str]
        PMC IDs *with or without* the “PMC” prefix (e.g. ``["PMC123", "456"]``).
    api_key : str | None, optional
        NCBI API key – lifts the personal rate-limit to ≈10 req s⁻¹.
    batch_size : int, default 200
        IDs per EFetch request (PMC ceiling = 10 000).
    timeout : int, default 20 s
        Socket timeout for each HTTP request.
    max_retries : int, default 3
        Attempts per batch on HTTP 429 or 5xx before giving up.
    delay : float, default 0.34 s
        Base pause between retries (doubles each attempt).

    Returns
    -------
    pandas.DataFrame
        Columns
        --------
        ``pmcid``   | string  
        ``fullXML`` | string  (entire `<article>` subtree or ``"N/A"``)

        Every PMC ID you supplied is represented exactly once—even if the
        record is missing, withdrawn, or the request fails.

    Example
    -------
    >>> df_xml = get_pmc_full_xml(["PMC9054321", "9054322"])
    >>> df_xml.loc[0, "fullXML"][:500]   # first 500 chars
    '<article article-type="research-article" ...'
    """
    # ── Guard clause ────────────────────────────────────────────
    if not pmcids:
        return pd.DataFrame(columns=["pmcid", "fullXML"]).astype("string")

    # Normalise IDs → keep “PMC” prefix for fetching & output
    norm_ids = [pid if str(pid).upper().startswith("PMC") else f"PMC{pid}" for pid in pmcids]

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    session = requests.Session()
    records: list[dict] = []

    total_batches = ceil(len(norm_ids) / batch_size)

    # ── Main loop ───────────────────────────────────────────────
    for b_idx in range(total_batches):
        chunk = norm_ids[b_idx * batch_size : (b_idx + 1) * batch_size]
        params = {"db": "pmc", "retmode": "xml", "id": ",".join(chunk)}
        if api_key:
            params["api_key"] = api_key

        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(base_url, params=params, timeout=timeout)
                response.raise_for_status()
                break
            except (HTTPError, RequestException) as exc:
                status = getattr(exc.response, "status_code", None)
                if status and (status == 429 or 500 <= status < 600) and attempt < max_retries:
                    wait = delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"Batch {b_idx+1}: HTTP {status}; retry {attempt}/{max_retries} in {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"Batch {b_idx+1} failed: {exc}")
                break

        if response is None or not response.ok:
            # give every ID in the chunk a placeholder row
            records.extend({"pmcid": cid, "fullXML": "N/A"} for cid in chunk)
            continue

        # ── Parse / strip namespace ─────────────────────────────
        try:
            root = ET.fromstring(_strip_default_ns(response.content))
        except ET.ParseError as e:
            logger.error(f"XML parse error in batch {b_idx+1}: {e}")
            records.extend({"pmcid": cid, "fullXML": "N/A"} for cid in chunk)
            time.sleep(delay)
            continue

        # EFetch returns multiple <article> elements in one payload
        seen = set()
        for art in root.findall(".//article"):
            pmcid_text = art.findtext('.//article-id[@pub-id-type="pmc"]') or "N/A"
            xml_str = ET.tostring(art, encoding="unicode")
            records.append({"pmcid": pmcid_text, "fullXML": xml_str})
            seen.add(pmcid_text)

        # Any IDs not returned (e.g., embargoed or withdrawn) → placeholder
        for cid in chunk:
            if cid not in seen:
                records.append({"pmcid": cid, "fullXML": "N/A"})

        time.sleep(delay)

    return pd.DataFrame(records).astype("string")
    
    
    
def get_pmc_html_text(
    pmcids: List[str],
    *,
    timeout: int = 20,
    max_retries: int = 3,
    delay: float = 0.5,
) -> pd.DataFrame:
    """
    ------------------------------------------------------------------------
    Download the **flat-HTML** body of any number of PubMed Central articles.
    ------------------------------------------------------------------------

    Parameters
    ----------
    pmcids : list[str]
        PMC IDs *with or without* the ``"PMC"`` prefix.  Duplicates are
        tolerated; each ID is represented exactly once in the output.
    timeout : int, default 20 s
        Socket timeout for each HTTP request.
    max_retries : int, default 3
        Attempts per article on HTTP 429 / 5xx before giving up.
    delay : float, default 0.5 s
        Base pause between retries (multiplied by 2**attempt).

    Returns
    -------
    pandas.DataFrame
        Columns
        --------
        ``pmcid``      | string  
        ``htmlText``   | string   (raw, cleaned HTML **as text**; ``"N/A"`` on failure)
        ``scrapeMsg``  | string   (empty on success, diagnostic message on failure)

        The frame always contains one row per requested PMCID, in the order
        they were supplied.
    """
    # ── Guard clause ────────────────────────────────────────────
    if not pmcids:
        return pd.DataFrame(columns=["pmcid", "htmlText", "scrapeMsg"]).astype("string")

    norm_ids = [pid if str(pid).upper().startswith("PMC") else f"PMC{pid}" for pid in pmcids]
    records = []

    base_tpl = "https://pmc.ncbi.nlm.nih.gov/articles/{pid}/?format=flat"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; PubMedCrawler/1.0; "
            "+https://github.com/you/yourrepo)"
        )
    }

    # ── Main loop over individual IDs ───────────────────────────
    for pid in norm_ids:
        url = base_tpl.format(pid=pid)
        html_text: str | None = None
        msg = ""

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)
                if resp.status_code in (403, 429) and attempt < max_retries:
                    wait = delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"{pid}: HTTP {resp.status_code}; retry {attempt}/{max_retries} in {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                # Prefer the content under #maincontent; fall back to full doc
                main = soup.find(id="maincontent") or soup
                # Keep HTML (with basic cleanup), not plain text:
                #   – drop <script>, <style>, navigation junk
                for tag in main.find_all(["script", "style", "nav", "footer", "aside"]):
                    tag.decompose()

                html_text = str(main)     # raw HTML as str
                break  # success – leave retry loop

            except (HTTPError, RequestException) as exc:
                msg = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries:
                    wait = delay * (2 ** (attempt - 1))
                    time.sleep(wait)
                    continue
                logger.error(f"{pid}: giving up after {max_retries} attempts – {msg}")
                html_text = None
            except Exception as exc:      # BeautifulSoup / unexpected
                msg = f"{type(exc).__name__}: {exc}"
                logger.error(f"{pid}: parsing error – {msg}")
                html_text = None
            finally:
                # avoid flooding PMC with rapid requests
                time.sleep(0.1)

        records.append({
            "pmcid": pid,
            "htmlText": html_text or "N/A",
            "scrapeMsg": msg,
        })

    return pd.DataFrame(records).astype("string")
    
    
    

def get_pmc_full_text(
    pmcids: List[str] | str,
    *,
    xml_fallback_min_chars: int = 2_000,
    timeout: int = 20
) -> dict[str, str]:
    """
    ------------------------------------------------------------------------
    Retrieve **plain full-text** for one or many PMCIDs.
       1) try the “flat” HTML view,  
       2) fall back to the JATS `<body>` from EFetch.
    ------------------------------------------------------------------------

    Parameters
    ----------
    pmcids : list[str] | str
        A single PMCID or an iterable of them (with or without "PMC" prefix).
    xml_fallback_min_chars : int, default 2 000
        If the flat view yields < min_chars characters, the function will
        attempt the XML route and keep whichever version is longer.
    timeout : int, default 20 s
        Socket timeout for each HTTP request.

    Returns
    -------
    dict[str, str]
        Mapping **pmcid → plain text**  
        (value is "N/A" when both attempts fail).

    Notes
    -----
    *   Designed for **few-at-a-time** access.  For bulk work use the OA
        Web Service tarballs instead.
    *   Keeps rate-limiting polite (100 ms sleep per request).
    """

    # normalise to list
    if isinstance(pmcids, str):
        pmcids = [pmcids]

    flat_tpl = "https://pmc.ncbi.nlm.nih.gov/articles/{pid}/?format=flat"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; PubMedCrawler/1.1; "
            "+https://github.com/you/yourrepo)"
        )
    }

    out: dict[str, str] = {}

    for raw_id in pmcids:
        pid = raw_id if str(raw_id).upper().startswith("PMC") else f"PMC{raw_id}"
        text = ""         # what we'll eventually keep
        success = False

        # ── step 1: flat HTML ──────────────────────────────────
        try:
            r = requests.get(flat_tpl.format(pid=pid), headers=headers, timeout=timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            main = soup.find(id="maincontent") or soup
            flat_text = " ".join(
                p.get_text(" ", strip=True) for p in main.find_all("p")
            )
            text = flat_text.strip()
            success = len(text) >= xml_fallback_min_chars
            if success:
                logger.info(f"{pid}: used flat HTML ({len(text):,} chars)")
        except Exception as exc:
            logger.warning(f"{pid}: flat view failed – {exc}")

        # ── step 2: XML fallback if needed ─────────────────────
        if not success:
            try:
                params = {"db": "pmc", "id": pid, "retmode": "xml"}
                r = requests.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params=params,
                    timeout=timeout,
                )
                r.raise_for_status()
                # ↓↓↓ parse AFTER stripping the default namespace
                root = ET.fromstring(_strip_default_ns(r.content))
                body = root.find(".//article/body")
                xml_text = (
                    ET.tostring(body, encoding="unicode", method="text").strip()
                    if body is not None
                    else ""
                )
                if len(xml_text) > len(text):
                    text = xml_text
                logger.info(f"{pid}: XML fallback used ({len(text):,} chars)")
            except Exception as exc:
                logger.error(f"{pid}: XML fallback failed – {exc}")


        out[pid] = text or "N/A"
        time.sleep(0.1)        # courtesy delay

    return out
    
    
def _scrape_pmc_standard_html(pmcid: str, *, timeout: int = 20) -> str:
    """
    Fetch the *regular* PMC HTML (not the `?format=flat` view) and return
    plain text.  Used only when both XML and flat-HTML versions are tiny.
    """
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    headers = {"User-Agent": "Mozilla/5.0 (PubMedCrawler/2.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # drop nav / scripts
        for tag in soup.find_all(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()
        return soup.get_text(" ", strip=True)
    except Exception as exc:
        logger.warning(f"{pmcid}: standard HTML scrape failed – {exc}")
        return "N/A"
        


##############################################################################
#  Main workflow                                                             #
##############################################################################
def fetch_pubmed_fulltexts(
    query: str,
    *,
    api_key: str | None = None,
    retmax: int = 2_000,
    min_fulltext_chars: int = 2_000
) -> pd.DataFrame:
    """
    ------------------------------------------------------------------------
    ONE-STOP WORKFLOW
    ------------------------------------------------------------------------
    1.  Search PubMed ➜ PMIDs.
    2.  Map PMIDs ➜ {0‒∞} PMCIDs.
    3.  Pull high-level PubMed metadata (title, abstract, …).
    4.  For *every* PMCID …
          a) JATS XML ➜ plain text
          b) flat-HTML ➜ plain text
          c) if both (a) & (b) < min_fulltext_chars ➜ scrape the
             regular article HTML as a last resort.
    5.  Return a tidy DataFrame with **one row per (PMID, PMCID)** pair:

        pmid | …metadata… | pmcid | xmlText | flatHtmlText | webScrapeText
    """
    # ── 1) SEARCH ──────────────────────────────────────────────
    pmids: List[str] = get_pmid_from_pubmed(
        query, retmax=retmax, api_key=api_key
    )
    if not pmids:
        logger.info("No PMIDs found – returning empty DataFrame")
        return pd.DataFrame()

    # ── 2) MAPPING ─────────────────────────────────────────────
    map_df = map_pmids_to_pmcids(
        pmids, api_key=api_key
    )  # → pmid, pmcid  (≥1 row / PMID, duplicates possible)

    # ── 3) METADATA ────────────────────────────────────────────
    meta_df = get_pubmed_metadata(pmids, api_key=api_key)

    # ── 4) FULL-TEXTS (XML & flat) ─────────────────────────────
    pmcids: List[str] = map_df["pmcid"].dropna().unique().tolist()
    xml_df  = get_pmc_full_xml(pmcids, api_key=api_key)            # pmcid, fullXML
    flat_df = get_pmc_html_text(pmcids)                            # pmcid, htmlText

    # convert JATS XML → plain text
    def xml_to_text(xml_str: str) -> str:
        """
        Helper for fetch_pubmed_fulltexts:
        take the <article> subtree as a string,
        return plain text from <body> or "N/A".
        """
        if xml_str == "N/A":
            return "N/A"
        try:
            elem = ET.fromstring(_strip_default_ns(xml_str.encode()))
            body = elem.find(".//body")
            return (
                ET.tostring(body, encoding="unicode", method="text").strip()
                if body is not None else "N/A"
            )
        except Exception:
            return "N/A"


    xml_df["xmlText"] = xml_df["fullXML"].apply(xml_to_text)
    xml_df = xml_df.drop(columns="fullXML")

    flat_df = flat_df.rename(columns={"htmlText": "flatHtmlText", "scrapeMsg": "flatMsg"})

    # ── 5) INITIAL MERGE (pmcid-level) ─────────────────────────
    pmcid_level = (
        xml_df.merge(flat_df, on="pmcid", how="outer")
              .fillna("N/A")
    )

    # ── 6) SECOND-LEVEL SCRAPE where needed ───────────────────
    need_scrape = (
        (pmcid_level["xmlText"].str.len() < min_fulltext_chars) &
        (pmcid_level["flatHtmlText"].str.len() < min_fulltext_chars)
    )
    if need_scrape.any():
        to_scrape = pmcid_level.loc[need_scrape, "pmcid"].tolist()
        logger.info(f"Additional standard-HTML scrape for {len(to_scrape)} PMCIDs")
        scrape_texts: Dict[str, str] = {
            pid: _scrape_pmc_standard_html(pid) for pid in to_scrape
        }
        pmcid_level["webScrapeText"] = pmcid_level["pmcid"].map(scrape_texts).fillna("N/A")
    else:
        pmcid_level["webScrapeText"] = "N/A"

    # even where we scraped, keep the other two text variants
    # (caller can decide which one to trust later)
    # ── 7) FINAL JOIN TO PMID MAP ──────────────────────────────
    wide = (
        map_df.merge(meta_df, on="pmid", how="left")
              .merge(pmcid_level, on="pmcid", how="left")
              .fillna("N/A")
    )

    # tidy up column order – pmid → metadata → pmcid → texts
    front = ["pmid"]
    meta_cols = [c for c in meta_df.columns if c != "pmid"]
    mid  = ["pmcid"]
    text_cols = ["xmlText", "flatHtmlText", "webScrapeText"]
    ordered_cols = front + meta_cols + mid + text_cols
    wide = wide[ordered_cols]

    return wide.astype("string")