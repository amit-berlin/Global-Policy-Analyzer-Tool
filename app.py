# app.py
import streamlit as st
import numpy as np
import pandas as pd
import re
import time
import json
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
try:
    import fitz  # PyMuPDF
    PDF_LIB = "pymupdf"
except Exception:
    from PyPDF2 import PdfReader
    PDF_LIB = "pypdf2"

st.set_page_config(page_title="2047 National Plan Think Tank — Agentic AI", layout="wide")

# ---------------------
# CSS for nicer cards
# ---------------------
st.markdown(
    """
    <style>
    .agent-card{
      border-radius:12px;
      padding:14px;
      margin-bottom:12px;
      background:linear-gradient(90deg, rgba(255,255,255,0.9), rgba(245,248,255,0.9));
      box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    .agent-title{font-weight:700; font-size:16px;}
    .small-muted{color:#6b7280; font-size:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------
# Utilities
# ---------------------
def safe_text(x: str) -> str:
    return (x or "").replace("\x00", " ").strip()

def extract_pdf_text(uploaded_file):
    # returns all text in the PDF, best effort
    if PDF_LIB == "pymupdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        pages = [safe_text(page.get_text()) for page in doc]
        return "\n".join(pages)
    else:
        reader = PdfReader(uploaded_file)
        pages = []
        for p in reader.pages:
            try:
                pages.append(safe_text(p.extract_text()))
            except Exception:
                pages.append("")
        return "\n".join(pages)

def chunk_text(text, chunk_size_words=250, overlap_words=50):
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    N = len(words)
    while i < N:
        chunk = words[i:i+chunk_size_words]
        chunks.append(" ".join(chunk))
        i += chunk_size_words - overlap_words
    return chunks

def top_keywords(texts, top_n=12):
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=1)
    X = vec.fit_transform(texts)
    means = np.asarray(X.mean(axis=0)).ravel()
    idx = means.argsort()[::-1][:top_n]
    return [(vec.get_feature_names_out()[i], float(means[i])) for i in idx]

def tiny_sentences(text):
    s = re.split(r'(?<=[.!?])\s+', text)
    return [x.strip() for x in s if x.strip()]

def tokens(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def compose_markdown(title, sections):
    md = [f"# {title}\n"]
    for sec_title, body in sections:
        md.append(f"## {sec_title}\n\n{body}\n")
    return "\n".join(md)

# ---------------------
# Sidebar Inputs
# ---------------------
st.sidebar.title("📌 2047 Plan Think Tank — Inputs")
st.sidebar.markdown("Upload policy docs, paste 10 national problems, set KPIs & budget. Agents will collaborate to produce a draft plan.")
policy_file = st.sidebar.file_uploader("Upload policy PDF (optional)", type=["pdf"])
problems_raw = st.sidebar.text_area("List up to 10 national problems (one per line)", height=200,
                                   value=(
                                       "Low agricultural productivity in semi-arid districts\n"
                                       "High urban unemployment among youth\n"
                                       "Intermittent potable water supply in peri-urban wards\n"
                                       "Chronic malnutrition pockets in tribal blocks\n"
                                       "Low female labour-force participation\n"
                                       "Digital divide in last-mile education\n"
                                       "Underperformance of micro-enterprises\n"
                                       "Slow grievance redressal in social assistance\n"
                                       "Infrastructure bottlenecks in rural roads\n"
                                       "Low quality of primary healthcare"
                                   ))
total_budget_cr = st.sidebar.number_input("Total Program Budget (Cr ₹)", min_value=10, value=1000)
target_year = st.sidebar.number_input("Target year", min_value=2030, max_value=2050, value=2047)
run_button = st.sidebar.button("Run Think Tank")

# ---------------------
# Page header
# ---------------------
st.title("🇮🇳 2047 National Development Plan — Agentic Policy Think Tank (MVP)")
st.markdown(
    "A lightweight, explainable think-tank pipeline that ingests policy, crowdsourced problems, "
    "and generates a draft national plan outline with simulations, budgeting, and a RAG-based QA assistant."
)
st.markdown("---")

# ---------------------
# Default baseline KPIs (demo)
# ---------------------
col1, col2, col3, col4 = st.columns(4)
current_year = datetime.now().year
kpi_gdp = col1.metric("GDP Growth (current %)", "5.5", delta="+0.3")
kpi_poverty = col2.metric("Poverty Rate (%)", "20.7", delta="-0.5")
kpi_literacy = col3.metric("Literacy Rate (%)", "77.7", delta="+0.6")
kpi_health = col4.metric("IMR (per 1000 live births)", "28", delta="-1")

# ---------------------
# Run pipeline
# ---------------------
if run_button:
    # Normalize problems
    problems = [p.strip() for p in problems_raw.splitlines() if p.strip()]
    problems = problems[:10]
    if not problems:
        st.error("Please enter at least one national problem in the sidebar.")
        st.stop()

    # AGENT 1: Policy Ingestor & Indexer
    with st.container():
        st.markdown('<div class="agent-card"><div class="agent-title">🤖 Agent-1: Policy Ingestor & Indexer</div>'
                    '<div class="small-muted">Reads documents and prepares retrieval index + quick stats</div></div>',
                    unsafe_allow_html=True)
        with st.spinner("Agent-1: ingesting..."):
            text = ""
            if policy_file:
                try:
                    text = extract_pdf_text(policy_file)
                except Exception as e:
                    st.warning("PDF reading failed — continuing with problems list only.")
            else:
                # fallback sample text for demo: synthesize a compact national vision paragraph
                text = ("A resilient India by 2047: inclusive growth through agriculture modernization, "
                        "industrial transformation, human capital development, universal digital access, "
                        "climate-resilient infrastructure, and transparent governance.")
            # build chunks
            chunks = chunk_text(text, chunk_size_words=220, overlap_words=40) or [" ".join(problems)]
            num_chunks = len(chunks)
            top_kw = top_keywords(chunks + problems, top_n=12)
            time.sleep(0.4)
        st.write(f"**Ingested chunks:** {num_chunks}")
        st.write("**Top signals / keywords:**")
        st.table(pd.DataFrame(top_kw, columns=["keyword", "score"]).assign(score=lambda d: d.score.round(3)))

    # AGENT 2: Thematizer & Knowledge Graph
    with st.container():
        st.markdown('<div class="agent-card"><div class="agent-title">🧭 Agent-2: Thematizer & Knowledge Graph Builder</div>'
                    '<div class="small-muted">Discovers themes, clusters problems & builds an explainable graph</div></div>',
                    unsafe_allow_html=True)
        with st.spinner("Agent-2: discovering themes..."):
            # topic modeling with NMF on problems+chunks
            corpus = problems + chunks
            n_topics = min(3, max(1, len(problems)//3))
            try:
                n_topics = max(1, n_topics)
                vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=1)
                X = vec.fit_transform(corpus)
                nmf = NMF(n_components=n_topics, random_state=42, init="nndsvd", max_iter=300)
                W = nmf.fit_transform(X)
                H = nmf.components_
                vocab = np.array(vec.get_feature_names_out())
                topics = [[vocab[idx] for idx in H[k].argsort()[::-1][:6]] for k in range(n_topics)]
            except Exception:
                topics = [["governance","infrastructure","human capital"]]
            # clustering problems
            try:
                groups, labels = {}, {}
                km = KMeans(n_clusters=min(3, max(1, len(problems))), n_init=10, random_state=42)
                vec2 = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
                Xp = vec2.fit_transform(problems)
                lbls = km.fit_predict(Xp)
                for i,l in enumerate(lbls):
                    groups.setdefault(int(l), []).append(problems[i])
            except Exception:
                groups = {0: problems}
            # build a small graph: problem -> topic -> signal(word)
            G = nx.DiGraph()
            for i,p in enumerate(problems):
                G.add_node(f"P{i+1}", label=f"P{i+1}", title=p, type="problem")
            for t_i, twords in enumerate(topics):
                G.add_node(f"T{t_i+1}", label=f"T{t_i+1}", title=", ".join(twords), type="topic")
            for i in range(len(problems)):
                tnode = f"T{(i % len(topics))+1}"
                G.add_edge(f"P{i+1}", tnode)
            # add top keyword nodes
            for kw,_ in top_kw[:6]:
                G.add_node(f"K_{kw}", label=kw, title=kw, type="keyword")
                # connect to one topic
                G.add_edge(f"T{((hash(kw) % len(topics))+1)}", f"K_{kw}")
            time.sleep(0.5)
        # show topics & clusters
        st.markdown("**Discovered themes:**")
        for idx, t in enumerate(topics, 1):
            st.write(f"**T{idx}** — {', '.join(t[:6])}")
        st.markdown("**Problem clusters:**")
        for c, issues in groups.items():
            st.write(f"Cluster {c+1}:")
            for i in issues: st.write(f"- {i}")
        # draw graph
        fig, ax = plt.subplots(figsize=(8,4))
        pos = nx.spring_layout(G, k=0.8, seed=42)
        colors = []
        labels = {}
        for n, d in G.nodes(data=True):
            labels[n] = d.get("title", n)[:40]
            t = d.get("type", "")
            if t=="problem": colors.append("#ffcccb")
            elif t=="topic": colors.append("#cfe9ff")
            else: colors.append("#eafbe7")
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=700, ax=ax)
        nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)
        ax.axis("off")
        st.pyplot(fig)

    # AGENT 3: RAG QA Agent chain (Retriever -> Evidence -> Exact Match -> Composer)
    with st.container():
        st.markdown('<div class="agent-card"><div class="agent-title">📚 Agent-3: RAG QA Assistant (Zero-hallucination)</div>'
                    '<div class="small-muted">Ask questions about the vision/proposals — agents will show retrieval, evidence & final extractive answer</div></div>',
                    unsafe_allow_html=True)
        # build TF-IDF index over chunks & problems
        corpus_rag = chunks + problems
        rag_vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=1)
        X_rag = rag_vec.fit_transform(corpus_rag)
        # QA widget
        q = st.text_input("Ask a question about the 2047 plan (RAG):", "")
        if q:
            with st.spinner("Agent-3.1 Retriever: finding relevant context..."):
                qv = rag_vec.transform([q])
                sims = cosine_similarity(qv, X_rag).flatten()
                topk = np.argsort(-sims)[:3]
                retrieved = [{"idx": int(i), "score": float(sims[i]), "text": corpus_rag[i]} for i in topk]
                time.sleep(0.3)
            st.markdown("**Agent-3.1 (Retriever) — top contexts:**")
            for r in retrieved:
                st.write(f"- (score {r['score']:.3f}) {r['text'][:280]}{'...' if len(r['text'])>280 else ''}")
            # evidence selection
            with st.spinner("Agent-3.2 Evidence Selector: sentence-level matching..."):
                evidence = []
                qv_sent = qv
                for r in retrieved:
                    sents = tiny_sentences(r['text'])
                    if not sents:
                        continue
                    S = rag_vec.transform(sents)
                    sims_s = cosine_similarity(qv_sent, S).flatten()
                    best_i = int(np.argmax(sims_s))
                    evidence.append({"chunk_idx": r['idx'], "sentence": sents[best_i], "sim": float(sims_s[best_i])})
                time.sleep(0.3)
            st.markdown("**Agent-3.2 (Evidence) — best sentences:**")
            for e in evidence:
                st.write(f"- (sim {e['sim']:.3f}) {e['sentence']}")
            # exact token match (Agent-3.3)
            with st.spinner("Agent-3.3 Exact Token Checker: strict matches..."):
                q_toks = set(tokens(q))
                exact_lines = []
                for e in evidence:
                    line_toks = set(tokens(e['sentence']))
                    common = q_toks.intersection(line_toks)
                    if len(common) >= max(1, min(3, len(q_toks))):  # len threshold relaxed
                        exact_lines.append({"chunk": e['chunk_idx'], "line": e['sentence'], "common": list(common), "sim": e['sim']})
                time.sleep(0.2)
            st.markdown("**Agent-3.3 (Exact token evidence):**")
            if exact_lines:
                for el in exact_lines:
                    st.success(f"Exact match (chunk {el['chunk']}): {el['line']} • tokens: {', '.join(el['common'])}")
            else:
                st.info("No strict exact-token match found; will produce extractive answer with caution.")
            # composer (Agent-3.4)
            with st.spinner("Agent-3.4 Composer: building extractive final answer..."):
                # pick top unique sentences by sim
                evidence = sorted(evidence, key=lambda x: -x['sim'])
                picked_sents = []
                seen = set()
                for e in evidence:
                    s = e['sentence'].strip()
                    if s and s not in seen:
                        picked_sents.append(s)
                        seen.add(s)
                    if len(picked_sents) >= 3:
                        break
                final_answer = " ".join(picked_sents) if picked_sents else "Insufficient evidence in ingested documents to answer the query."
                # confidence heuristic: max sim and token overlap fraction
                max_sim = max([e['sim'] for e in evidence]) if evidence else 0.0
                tok_cov = (len(set(tokens(final_answer)).intersection(q_toks)) / max(1, len(q_toks)))
                confidence = float(np.clip(0.5*max_sim + 0.5*tok_cov, 0.0, 1.0))
                time.sleep(0.2)
            st.markdown("**Agent-3.4 (Final extractive answer)**")
            if confidence >= 0.18:
                st.success(final_answer)
                st.caption(f"Confidence: {confidence:.2f} (heuristic — higher is better)")
            else:
                st.warning("I abstain: evidence insufficient or low confidence. Consider uploading more documents or rephrasing.")
                st.caption(f"Confidence: {confidence:.2f}")

    # AGENT 4: Impact Simulator (scenarios to 2047)
    with st.container():
        st.markdown('<div class="agent-card"><div class="agent-title">📈 Agent-4: Impact Simulator & Scenarios</div>'
                    '<div class="small-muted">Simulates KPI trajectories under multiple scenarios (baseline/optimistic/policy-accelerated)</div></div>',
                    unsafe_allow_html=True)
        with st.spinner("Agent-4: simulating to target year..."):
            # basic demo KPIs
            years = np.arange(current_year, target_year+1)
            n = len(years)
            # baseline parameters (mock)
            gdp_curr = 5.5
            gdp_baseline = gdp_curr + np.linspace(0, 1.0, n)  # modest growth
            # policy-accelerated: add effect proportional to interventions count
            policy_effect = min(0.02 * max(1, len(problems)), 0.12)  # up to +12% cumulative effect
            gdp_policy = gdp_baseline + np.linspace(0, policy_effect*5, n)  # scaled
            # pessimistic scenario
            gdp_pess = gdp_baseline - np.linspace(0, 0.8, n)
            df_sim = pd.DataFrame({"year": years, "baseline": gdp_baseline, "policy": gdp_policy, "pessimistic": gdp_pess})
            time.sleep(0.5)
        st.markdown("**Projected GDP growth scenarios (%)**")
        st.line_chart(df_sim.set_index("year"))
        # show small KPI cards for 2047
        cA, cB, cC = st.columns(3)
        cA.metric(f"GDP Growth (2047) — Policy", f"{df_sim['policy'].iloc[-1]:.2f}%")
        cB.metric(f"Poverty Rate (demo)", f"{max(3.0, 20.7 - (len(problems)*0.8)):.1f}%")
        cC.metric("Projected Uptime of key portals", "99.3%")

    # AGENT 5: Budget Optimizer & Policy Composer
    with st.container():
        st.markdown('<div class="agent-card"><div class="agent-title">💼 Agent-5: Budget Optimizer & Policy Composer</div>'
                    '<div class="small-muted">Allocates budget to priority interventions and drafts an implementation plan</div></div>',
                    unsafe_allow_html=True)
        with st.spinner("Agent-5: proposing interventions and budget..."):
            # simple interventions mapping from keywords
            # reuse top keywords to map to some interventions (lightweight rule-based)
            suggested_interventions = []
            for kw, _ in top_kw[:12]:
                if "education" in kw or "school" in kw or "skill" in kw:
                    suggested_interventions.append("National Skill & School Digitalization Program")
                elif "health" in kw or "malnutrition" in kw:
                    suggested_interventions.append("District Health Strengthening & Supply Chain")
                elif "water" in kw:
                    suggested_interventions.append("Urban Water Quality & NRW Reduction")
                elif "farmer" in kw or "agriculture" in kw:
                    suggested_interventions.append("Agri Productivity & Market Access")
                elif "portal" in kw or "digital" in kw:
                    suggested_interventions.append("Digital Access & Last-mile Connectivity")
            # fallback
            if not suggested_interventions:
                suggested_interventions = ["Baseline Rapid Response Grants", "Field Monitoring Pilots"]
            # dedupe
            suggested_interventions = list(dict.fromkeys(suggested_interventions))[:6]
            # allocate budget proportionally by intervention importance (equal weight for demo)
            n_int = len(suggested_interventions)
            per_share = round(total_budget_cr / n_int, 2) if n_int > 0 else 0
            budget_table = pd.DataFrame({
                "Intervention": suggested_interventions,
                "Allocated Budget (Cr ₹)": [per_share]*n_int,
                "Timeline": ["0-90 days pilot" if i<2 else "90-365 days scale" for i in range(n_int)]
            })
            # simple implementation steps
            impl_steps = [
                "Phase 0 (0-30 days): Baseline survey, stakeholder mapping, governance setup",
                "Phase 1 (30-120 days): 3-district pilots, weekly dashboards, weekly reviews",
                "Phase 2 (120-365 days): State-wide scale for successful pilots, monthly KPIs",
                "Phase 3 (Year 2-5): Nationwide rollout where pilot succeeded, institutionalize M&E"
            ]
            time.sleep(0.4)
        st.markdown("**Suggested interventions & allocation (illustrative)**")
        st.table(budget_table)
        # pie chart
        fig2, ax2 = plt.subplots(figsize=(5,3))
        ax2.pie(budget_table["Allocated Budget (Cr ₹)"], labels=budget_table["Intervention"], autopct="%1.1f%%", startangle=140)
        ax2.axis("equal")
        st.pyplot(fig2)

        # Compose final markdown policy draft
        sections = [
            ("Executive Summary", "A resilient, inclusive and digital India by 2047. Focus on agriculture, health, skills, infrastructure, and governance."),
            ("Problems Addressed", "\n".join([f"- {p}" for p in problems])),
            ("Priority Interventions", "\n".join([f"- {r}" for r in suggested_interventions])),
            ("Budget (illustrative)", "\n".join([f"- {row['Intervention']}: ₹{row['Allocated Budget (Cr ₹)']} Cr" for _,row in budget_table.iterrows()])),
            ("Implementation Roadmap", "\n".join([f"- {s}" for s in impl_steps])),
            ("Monitoring & M&E", "- Monthly KPI dashboards, quarterly independent audits, public transparency portals"),
            ("Risks & Mitigation", "- Data quality, change management, infra; mitigation: audits, capacity building, redundancy")
        ]
        draft_md = compose_markdown("Draft 2047 Plan Outline (MVP)", sections)

        st.markdown("#### Draft Policy Outline (Markdown)")
        st.code(draft_md, language="markdown")
        st.download_button("⬇️ Download Draft (Markdown)", draft_md, file_name="draft_2047_plan.md", mime="text/markdown")

    # Summarize endpoint (JSON)
    with st.expander("🔖 Full JSON package (downloadable)"):
        package = {
            "problems": problems,
            "top_keywords": top_kw,
            "topics": topics,
            "clusters": groups,
            "interventions": suggested_interventions,
            "budget": budget_table.to_dict(orient="records"),
            "draft_markdown": draft_md
        }
        st.json(package)
        st.download_button("⬇️ Download JSON Package", json.dumps(package, indent=2), file_name="draft_package.json", mime="application/json")

    st.success("Think Tank run complete — review Agent outputs above and download the draft policy.")
else:
    st.info("Enter inputs in the sidebar and click **Run Think Tank** to generate a 2047 policy draft.")
