#!/usr/bin/env python3
"""Ingest authentic verbatim transcripts from public archives into KLens samples/."""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
USER_AGENT = "KLens-Corpus-Ingest/1.0"


def fetch_bytes(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_text(url: str, timeout: int = 30) -> str:
    return fetch_bytes(url, timeout=timeout).decode("utf-8", errors="replace")


def fetch_json(url: str, timeout: int = 30) -> Any:
    return json.loads(fetch_text(url, timeout=timeout))


def bucket_turns(
    raw_turns: list[tuple[str, str]],
    num_rounds: int,
    min_chars: int = 12,
) -> list[dict[str, Any]]:
    filtered = [
        (sp.strip(), re.sub(r"\s+", " ", tx).strip())
        for sp, tx in raw_turns
        if sp.strip() and len(tx.strip()) >= min_chars
    ]
    total = len(filtered)
    out: list[dict[str, Any]] = []
    for idx, (speaker, text) in enumerate(filtered):
        rnd = min(num_rounds, 1 + (idx * num_rounds) // max(1, total))
        out.append({"round": rnd, "speaker": speaker, "text": text})
    return out


def save_dataset(
    filename: str,
    title: str,
    context: str,
    metadata: dict[str, Any],
    turns: list[dict[str, Any]],
) -> Path:
    path = SAMPLES_DIR / filename
    speakers = sorted({t["speaker"] for t in turns})
    rounds = sorted({t["round"] for t in turns})
    payload = {
        "title": title,
        "context": context,
        "metadata": {
            **metadata,
            "agents": len(speakers),
            "rounds": len(rounds),
            "total_turns": len(turns),
        },
        "turns": turns,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {filename}: agents={len(speakers)}, rounds={len(rounds)}, turns={len(turns)}")
    return path


def ingest_scotus_trump_v_us() -> None:
    url = "https://api.oyez.org/case_media/oral_argument_audio/25645"
    data = fetch_json(url)
    sections = data.get("transcript", {}).get("sections", [])
    raw_turns: list[tuple[str, str]] = []
    for sec in sections:
        for t in sec.get("turns", []):
            speaker = (t.get("speaker") or {}).get("name") or "Unknown"
            blocks = [b.get("text", "") for b in t.get("text_blocks", []) if b.get("text")]
            text = " ".join(blocks).strip()
            if text:
                raw_turns.append((speaker, text))

    turns = bucket_turns(raw_turns, num_rounds=10, min_chars=30)
    save_dataset(
        "scotus_trump_v_us_2024.json",
        "U.S. Supreme Court Oral Argument: Trump v. United States (2024)",
        "Verbatim Oyez transcript of No. 23-939 oral argument (April 25, 2024) on presidential criminal immunity across 10 chronological segments.",
        {
            "category": "institutional_proceeding",
            "institution": "Supreme Court of the United States",
            "case": "Trump v. United States, No. 23-939",
            "date": "2024-04-25",
            "source_url": url,
        },
        turns,
    )


def ingest_fomc_covid_mar2020() -> None:
    url = "https://www.federalreserve.gov/monetarypolicy/files/FOMC20200315meeting.pdf"
    pdf_bytes = fetch_bytes(url, timeout=45)
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        raw_txt = subprocess.check_output(["pdftotext", tmp.name, "-"]).decode(
            "utf-8", errors="replace"
        )

    speaker_re = re.compile(r"^((?:CHAIR|VICE CHAIR|MR\.|MS\.|GOVERNOR)\s+[A-Z][A-Z\-]+)\.\s*(.*)$")
    skip_re = re.compile(
        r"^(?:March 15, 2020\s+\d+\s+of\s+\d+|\d+\s+of\s+\d+|Meeting of the Federal Open Market Committee|"
        r"FINAL TRANSCRIPT|Authorized for Public Release)"
    )

    aliases = {"CHAIR WILLIAMS": "VICE CHAIR WILLIAMS"}
    raw_turns: list[tuple[str, str]] = []
    cur_sp: str | None = None
    cur_lines: list[str] = []

    for line in raw_txt.splitlines():
        s = re.sub(r"\bMarch 15, 2020\b|\b\d+\s+of\s+\d+\b", " ", line).strip()
        s = re.sub(r"\s+", " ", s)
        if not s or skip_re.match(s):
            continue
        m = speaker_re.match(s)
        if m:
            if cur_sp and cur_lines:
                raw_turns.append((cur_sp, " ".join(cur_lines)))
            sp = aliases.get(m.group(1), m.group(1)).title()
            cur_sp = sp
            cur_lines = [m.group(2).strip()] if m.group(2).strip() else []
        elif cur_sp is not None:
            cur_lines.append(s)

    if cur_sp and cur_lines:
        raw_turns.append((cur_sp, " ".join(cur_lines)))

    counts: dict[str, int] = {}
    for sp, _ in raw_turns:
        counts[sp] = counts.get(sp, 0) + 1
    filtered = [(sp, tx) for sp, tx in raw_turns if counts.get(sp, 0) >= 2]

    turns = bucket_turns(filtered, num_rounds=8, min_chars=15)
    save_dataset(
        "fomc_emergency_covid_mar2020.json",
        "Federal Open Market Committee Emergency Meeting (March 15, 2020)",
        "Verbatim Federal Reserve transcript of the Sunday COVID-19 emergency meeting cutting rates to the zero lower bound and launching quantitative easing.",
        {
            "category": "institutional_proceeding",
            "institution": "Federal Reserve FOMC",
            "date": "2020-03-15",
            "source_url": url,
        },
        turns,
    )


def ingest_senate_ai_oversight_2023() -> None:
    url = "https://www.govinfo.gov/content/pkg/CHRG-118shrg52483/html/CHRG-118shrg52483.htm"
    raw_html = fetch_text(url)
    m_pre = re.search(r"<pre>(.*?)</pre>", raw_html, re.S | re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "", m_pre.group(1) if m_pre else raw_html))

    speaker_re = re.compile(
        r"^\s{2,8}((?:Chairman|Senator|Mr\.|Ms\.|Dr\.)\s+[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)?)\.\s+(.*)$"
    )
    aliases = {"Dr. Matheny": "Mr. Matheny"}

    raw_turns: list[tuple[str, str]] = []
    cur_sp: str | None = None
    cur_lines: list[str] = []

    for line in text.splitlines():
        m = speaker_re.match(line)
        if m:
            if cur_sp and cur_lines:
                raw_turns.append((cur_sp, " ".join(cur_lines)))
            sp = aliases.get(m.group(1).strip(), m.group(1).strip())
            cur_sp = sp
            cur_lines = [m.group(2).strip()]
        elif cur_sp is not None:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("[The ", "OPENING STATEMENT OF", "PREPARED STATEMENT OF", "=== ")):
                raw_turns.append((cur_sp, " ".join(cur_lines)))
                cur_sp = None
                cur_lines = []
                continue
            cur_lines.append(stripped)

    if cur_sp and cur_lines:
        raw_turns.append((cur_sp, " ".join(cur_lines)))

    counts: dict[str, int] = {}
    for sp, _ in raw_turns:
        counts[sp] = counts.get(sp, 0) + 1
    filtered = [(sp, tx) for sp, tx in raw_turns if counts.get(sp, 0) >= 3]

    turns = bucket_turns(filtered, num_rounds=8, min_chars=20)
    save_dataset(
        "senate_ai_oversight_hearing_2023.json",
        "U.S. Senate Hearing: Artificial Intelligence Risks & Opportunities (2023)",
        "Verbatim GovInfo transcript of S. Hrg. 118-25 featuring Senators Peters, Blumenthal, Hassan, Johnson, Padilla, Rosen, Sinema and expert witnesses Matheny, Venkatasubramanian, and Givens.",
        {
            "category": "institutional_proceeding",
            "institution": "U.S. Senate Committee on Homeland Security and Governmental Affairs",
            "package_id": "CHRG-118shrg52483",
            "source_url": url,
        },
        turns,
    )


def _extract_ucsb_body(page_html: str) -> str:
    start = page_html.find("field-docs-content")
    if start == -1:
        raise ValueError("Missing field-docs-content container")
    end = page_html.find("</article>", start)
    return page_html[start:end] if end != -1 else page_html[start:]


def ingest_debate_kennedy_nixon_1960_full() -> None:
    url = "https://www.presidency.ucsb.edu/documents/presidential-debate-chicago"
    body = _extract_ucsb_body(fetch_text(url))
    plain = html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"</p>|<br\s*/?>", "\n", body)))

    speaker_re = re.compile(r"^([A-Za-z.\- ]{2,25}):\s+(.*)$")
    canonical = {
        "MR. SMITH": "Howard K. Smith",
        "MR. KENNEDY": "John F. Kennedy",
        "MR. NIXON": "Richard M. Nixon",
        "MR. NOVINS": "Stuart Novins",
        "MR. WARREN": "Charles Warren",
        "MR. VANOCUR": "Sander Vanocur",
        "MR. FLEMING": "Robert Fleming",
    }

    raw_turns: list[tuple[str, str]] = []
    cur_sp: str | None = None
    cur_lines: list[str] = []

    for line in plain.splitlines():
        s = line.strip()
        if not s:
            continue
        m = speaker_re.match(s)
        if m and m.group(1).strip().upper() in canonical:
            if cur_sp and cur_lines:
                raw_turns.append((cur_sp, " ".join(cur_lines)))
            cur_sp = canonical[m.group(1).strip().upper()]
            cur_lines = [m.group(2).strip()]
        elif cur_sp is not None:
            cur_lines.append(s)

    if cur_sp and cur_lines:
        raw_turns.append((cur_sp, " ".join(cur_lines)))

    turns = bucket_turns(raw_turns, num_rounds=8, min_chars=15)
    save_dataset(
        "debate_kennedy_nixon_1960_full.json",
        "First Kennedy-Nixon Presidential Debate (Chicago, 1960 — Full Verbatim Transcript)",
        "Complete verbatim American Presidency Project transcript of the September 26, 1960 broadcast debate across 8 chronological Q&A segments.",
        {
            "category": "historical_debate_full",
            "date": "1960-09-26",
            "location": "Chicago, Illinois",
            "source_url": url,
        },
        turns,
    )


def ingest_debate_bush_clinton_perot_1992_full() -> None:
    url = "https://www.presidency.ucsb.edu/documents/presidential-debate-st-louis"
    body = _extract_ucsb_body(fetch_text(url))
    paras = [
        html.unescape(re.sub(r"<[^>]+>", "", p)).strip()
        for p in re.findall(r"<p>(.*?)</p>", body, re.S)
    ]

    speaker_re = re.compile(
        r"^((?:President |Governor |Mr\. |Ms\. )[A-Z][a-zA-Z\-]+)\.\s+(.*)$"
    )
    canonical = {
        "Mr. Lehrer": "Jim Lehrer",
        "Mr. Perot": "Ross Perot",
        "Governor Clinton": "Bill Clinton",
        "President Bush": "George H.W. Bush",
        "Ms. Compton": "Ann Compton",
        "Mr. Vanocur": "Sander Vanocur",
        "Mr. Mashek": "John Mashek",
    }

    raw_turns: list[tuple[str, str]] = []
    cur_sp: str | None = None
    cur_lines: list[str] = []

    for para in paras:
        if not para:
            continue
        m = speaker_re.match(para)
        if m and m.group(1) in canonical:
            if cur_sp and cur_lines:
                raw_turns.append((cur_sp, " ".join(cur_lines)))
            cur_sp = canonical[m.group(1)]
            cur_lines = [m.group(2).strip()]
        elif cur_sp is not None:
            cur_lines.append(para)

    if cur_sp and cur_lines:
        raw_turns.append((cur_sp, " ".join(cur_lines)))

    turns = bucket_turns(raw_turns, num_rounds=10, min_chars=15)
    save_dataset(
        "debate_bush_clinton_perot_1992_full.json",
        "1992 St. Louis Presidential Debate: Bush, Clinton & Perot (Full Verbatim Transcript)",
        "Complete verbatim American Presidency Project transcript of the October 11, 1992 three-candidate debate across 10 chronological segments.",
        {
            "category": "historical_debate_full",
            "date": "1992-10-11",
            "location": "St. Louis, Missouri",
            "source_url": url,
        },
        turns,
    )


def _parse_werewolf_bench(url: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    txt = fetch_text(url)
    lines = txt.splitlines()
    ignore_tags = {"ACTION", "CONSENSUS", "WITCH", "ROUND", "WARNING", "VOTE"}

    meta: dict[str, str] = {}
    for l in lines[:12]:
        if ":" in l and not l.startswith("="):
            k, v = l.split(":", 1)
            meta[k.strip()] = v.strip()

    phase = "Election"
    cur_day = 0
    raw_events: list[tuple[str, str, str]] = []
    i = 0
    while i < len(lines):
        l = lines[i]
        s = l.strip()
        if "🌙 NIGHT" in s:
            m = re.search(r"NIGHT\s+(\d+)", s)
            cur_day = int(m.group(1)) if m else cur_day + 1
            phase = f"Night_{cur_day}"
        elif "☀️ DAY" in s:
            m = re.search(r"DAY\s+(\d+)", s)
            cur_day = int(m.group(1)) if m else cur_day + 1
            phase = f"Day_{cur_day}"

        m1 = re.match(r"^\s*🎤\s*([A-Z][a-z]+)\s*:\s*(.+)$", l)
        m2 = re.match(r"^\[([A-Z][a-z]+)\]\s+(.+)$", s)
        if m1:
            raw_events.append((phase, m1.group(1), m1.group(2).strip()))
        elif m2 and m2.group(1) not in ignore_tags and not m2.group(2).startswith("-"):
            sp = m2.group(1)
            body = [m2.group(2).strip()]
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()
                if nxt.startswith(("🤔", "[", "🌙", "☀️", "---", "===", "📊", "🎭", "💀", "🏁")):
                    break
                if nxt:
                    body.append(nxt)
                j += 1
            raw_events.append((phase, sp, " ".join(body)))
            i = j - 1
        i += 1

    # Assign cycles by phase: Election -> R1; Day_1 splits every set-repeat or survivor count -> R2..4; Day_2 -> R5..7
    turns: list[dict[str, Any]] = []
    phase_order: list[str] = []
    grouped: dict[str, list[tuple[str, str]]] = {}
    for ph, sp, tx in raw_events:
        if ph not in grouped:
            phase_order.append(ph)
            grouped[ph] = []
        grouped[ph].append((sp, tx))

    rnd = 0
    for ph in phase_order:
        items = grouped[ph]
        if ph == "Election":
            rnd += 1
            for sp, tx in items:
                turns.append({"round": rnd, "speaker": sp, "text": tx})
        else:
            seen: set[str] = set()
            rnd += 1
            for sp, tx in items:
                if sp in seen:
                    rnd += 1
                    seen.clear()
                seen.add(sp)
                turns.append({"round": rnd, "speaker": sp, "text": tx})
    return meta, turns


def ingest_werewolf_benchmarks() -> None:
    specs = [
        (
            "real_llm_werewolf_gpt5_vs_gemini.json",
            "https://raw.githubusercontent.com/Foaster-ai/Werewolf-bench/main/Game_35.txt",
            "Real Multi-LLM Social Deduction: GPT-5 Villagers vs Gemini-2.5-Pro Wolves (Game 35)",
            "Authentic Foaster-ai Werewolf-bench transcript (Public speeches only) showing wolf betrayal, Seer counter-claims, and village convergence across 7 deliberation phases.",
        ),
        (
            "real_llm_werewolf_wolves_win.json",
            "https://raw.githubusercontent.com/Foaster-ai/Werewolf-bench/main/Game_203.txt",
            "Real Multi-LLM Deception: Gemini-2.5-Flash Villagers vs Gemini-2.5-Pro Wolves (Game 203)",
            "Authentic Foaster-ai Werewolf-bench transcript (Public speeches only) where frontier wolf agents sustain deceptive coalitions through Game 203 victory.",
        ),
    ]
    for fn, url, title, ctx in specs:
        meta, turns = _parse_werewolf_bench(url)
        save_dataset(
            fn,
            title,
            ctx,
            {
                "category": "real_multi_llm_deliberation",
                "benchmark": "Foaster-ai/Werewolf-bench",
                "source_url": url,
                **{k.lower().replace(" ", "_"): v for k, v in meta.items()},
            },
            turns,
        )


def _clean_negotiation_text(raw: str) -> str:
    txt = re.sub(r"<SCRATCHPAD>.*?</SCRATCHPAD>", "", raw, flags=re.S).strip()
    if "<SCRATCHPAD>" in txt:
        m_deal = re.search(r"(<DEAL>.*</DEAL>.*)", txt, re.S)
        if m_deal:
            txt = m_deal.group(1).strip()
        else:
            lines = [
                l.strip()
                for l in txt.splitlines()
                if l.strip() and not l.strip().startswith("<SCRATCHPAD>")
            ]
            txt = " ".join(lines[-3:]) if lines else txt.replace("<SCRATCHPAD>", "").strip()
    txt = re.sub(r"^\s*I propose the following deal:\s*", "", txt, flags=re.I)
    txt = re.sub(r"</?DEAL>", "", txt)
    return re.sub(r"\s+", " ", txt).strip()


def ingest_llm_negotiations() -> None:
    specs = [
        (
            "real_llm_negotiation_gpt4_coop.json",
            "https://raw.githubusercontent.com/S-Abdelnabi/LLM-Deliberation/main/logs/all_coop_base_7player_gpt4/all_coop_7players/history19_28_45.json",
            "Real Multi-LLM Stakeholder Negotiation: 7 Cooperative GPT-4 Agents (30 Turns)",
            "Authentic NeurIPS LLM-Deliberation log of 7 GPT-4 city stakeholders (SportCo, Mayor, Union, Environmental League, Heritage, Tourism, Other cities) negotiating a multi-issue stadium deal across 10 chronological windows.",
            10,
            "cooperative_7player_gpt4",
        ),
        (
            "real_llm_negotiation_targeted_adv.json",
            "https://raw.githubusercontent.com/cmpatino/llm_negotiation/main/games_descriptions/base/output/gpt4o_mini/targeted_adv/row2/history13_37_53.json",
            "Real Multi-LLM Adversarial Negotiation: Targeted Sabotage Protocol (26 Turns)",
            "Authentic multi-agent LLM stakeholder negotiation log (GPT-4o-mini) with a targeted adversary attempting to derail multi-issue coalition consensus across 9 chronological windows.",
            9,
            "targeted_adversary_6player_gpt4o_mini",
        ),
    ]
    for fn, url, title, ctx, n_rounds, regime in specs:
        data = fetch_json(url)
        raw_turns: list[tuple[str, str]] = []
        for r in data.get("rounds", []):
            sp = r.get("agent", "").strip()
            txt = _clean_negotiation_text(r.get("public_answer") or r.get("full_answer") or "")
            if sp and txt:
                raw_turns.append((sp, txt))
        turns = bucket_turns(raw_turns, num_rounds=n_rounds, min_chars=8)
        save_dataset(
            fn,
            title,
            ctx,
            {
                "category": "real_multi_llm_deliberation",
                "regime": regime,
                "finished_rounds": data.get("finished_rounds"),
                "source_url": url,
            },
            turns,
        )


def main() -> None:
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    ingest_scotus_trump_v_us()
    ingest_fomc_covid_mar2020()
    ingest_senate_ai_oversight_2023()
    ingest_debate_kennedy_nixon_1960_full()
    ingest_debate_bush_clinton_perot_1992_full()
    ingest_werewolf_benchmarks()
    ingest_llm_negotiations()


if __name__ == "__main__":
    main()
