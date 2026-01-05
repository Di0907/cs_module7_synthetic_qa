import os, json, time, textwrap
import feedparser

def fetch_arxiv(query: str, max_results: int = 100):
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={query}&start=0&max_results={max_results}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    feed = feedparser.parse(url)
    papers = []
    for e in feed.entries:
        arxiv_id = e.id.split("/abs/")[-1]
        papers.append({
            "arxiv_id": arxiv_id,
            "title": getattr(e, "title", "").replace("\n", " ").strip(),
            "abstract": getattr(e, "summary", "").replace("\n", " ").strip(),
            "link": getattr(e, "link", ""),
        })
    return papers

def build_prompt(title: str, abstract: str) -> str:
    return textwrap.dedent(f"""
    Generate exactly 5 QA pairs from the abstract.

    Constraints:
    - All answers must be supported ONLY by the abstract.
    - Q1-Q4: normal academic QA.
    - Q5: MUST be a "misinterpretation + correction" example:
      * include a plausible but incorrect user interpretation
      * then correct it and answer correctly

    Output STRICT JSON:
    {{
      "qa": [
        {{"type":"normal","question":"...","answer":"..."}},
        ...
        {{"type":"misinterpretation_correction","misinterpretation":"...","question":"...","answer":"..."}}
      ]
    }}

    Title: {title}
    Abstract: {abstract}
    """).strip()

def openai_generate_qa(prompt: str):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role":"system","content":"You are a careful academic QA generator."},
            {"role":"user","content":prompt},
        ],
        temperature=0.4,
    )
    return json.loads(resp.choices[0].message.content)

def main():
    query = "cat:cs.CL"      
    max_papers = 100
    out_path = os.path.join("data", "dataset.jsonl")

    papers = fetch_arxiv(query, max_papers)

    rows = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for p in papers:
            if not p["abstract"]:
                continue
            prompt = build_prompt(p["title"], p["abstract"])
            data = openai_generate_qa(prompt)

            for item in data["qa"]:
                if item["type"] == "normal":
                    record = {
                        "instruction": "Answer the question using ONLY the provided abstract.",
                        "input": f"[TITLE]\n{p['title']}\n\n[ABSTRACT]\n{p['abstract']}\n\n[QUESTION]\n{item['question']}",
                        "output": item["answer"],
                        "meta": {"arxiv_id": p["arxiv_id"], "type":"normal", "link": p["link"]},
                    }
                else:
                    record = {
                        "instruction": "The user misunderstood the abstract. Correct them politely and answer using ONLY the abstract.",
                        "input": (
                            f"[TITLE]\n{p['title']}\n\n[ABSTRACT]\n{p['abstract']}\n\n"
                            f"[USER MISINTERPRETATION]\n{item.get('misinterpretation','')}\n\n"
                            f"[QUESTION]\n{item.get('question','')}"
                        ),
                        "output": item["answer"],
                        "meta": {"arxiv_id": p["arxiv_id"], "type":"misinterpretation_correction", "link": p["link"]},
                    }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows += 1

            time.sleep(0.4)  
    print(f"Saved {rows} lines to {out_path}")

if __name__ == "__main__":
    main()
