# GlobalEdge Brokerage — FINAL VS CODE AGENT IMPLEMENTATION PROMPT

> Paste this complete prompt into **VS Code Chat in Agent mode** with the extracted `GlobalEdge_RAG_VSCode_Workspace` folder open. This is an implementation task, not a request for another plan. Use the model/API access already available in this session for the approved online run path; do not use unsupported external paid services or ask for additional credentials during implementation.

You are the lead Python / RAG engineer and technical presentation editor finishing the **UT Austin/Great Learning Project 2: GlobalEdge Brokerage Financial Intelligence RAG** assignment using the **LOW-CODE** track. Build a reliable, reproducible, business-context-first solution and prepare it for a **single final interactive Google Colab session**. Do the implementation and validation in VS Code, using the session's approved API access when a live run is needed. Do not use the full-code notebook.

## 0. Begin by inspecting the actual workspace

Read these files completely and resolve any conflicts in favor of (1) documented explicit user approvals, (2) original course notebook and slide requirements, then (3) the reference drafts:

- `templates/UTAIGA_Project_2_Low_Code_Notebook.ipynb` — authoritative original course notebook: **94 cells**, 47 code and 47 Markdown, with required headings/order.
- `reference/UTAIGA_Project_2_Low_Code_Implemented.ipynb` — existing implementation draft, not automatically assumed runtime-correct.
- `templates/Project-2 Template.pptx` — original 11-slide Great Learning template.
- `reference/UTAIGA_Project_2_Presentation_Draft_PreRun.pptx` — 14-slide preliminary presentation; preserve template visual design and reuse useful content.
- `docs/UTAIGA_Project_2_Decision_Log.md` — comprehensive approvals, rationales, tradeoffs and current status.
- `docs/UTAIGA_Project_2_Offline_Preflight.md` and `docs/UTAIGA_Project_2_Colab_Run_Guide.md` — existing offline QA notes and execution guide; independently verify, do not trust success claims without tests.
- `input/global_news.csv`, `input/stock_price_details.csv`, `input/sec_filings_10k.txt`, `input/golden_benchmark_dataset.csv` — the **only allowed retrieval/evidence corpus** plus the separate evaluation benchmark. Never download outside financial or market sources to fill gaps.

The supplied notebook prose mistakenly names `all_prices_clean.csv` and `sec_filings.txt`. The actual provided source files are `stock_price_details.csv` and `sec_filings_10k.txt`. Correct references without changing section order. The supplied SEC text is **Microsoft FY2024 10-K only**, not Apple/Amazon/Alphabet filings nor 10-Qs. The golden set references some absent filings. Don't treat its `context`, `supporting_sources`, or `response` as retrievable evidence. This project handles *historical snapshots*, not a live market feed.

## 1. Nonnegotiable business objective

GlobalEdge brokers need **fast, accurate, dated, evidence-backed financial research**. Prioritize safety, company/date accuracy, abstentions for unsupported claims and clear auditability above verbosity or superficial benchmark score. Do not generate unsupported exact financial values, unsupported investment recommendations, current quotes, fabricated filing coverage, or made-up citations. Ground every claim in the actual retrieved corpus. When the question asks for current market information, explain the snapshot limitation; answer historical portions only where supported. Never say source citations *prove* an answer unless the cited excerpt actually supports it. The interface is research support, not personalized investment advice.

## 2. Approved choices — do not reopen these decisions

| Choice | Approved value |
|---|---|
| Development | **VS Code agent** (new workflow); final target **Google Colab**, use the approved live API access in this session for execution and validation when required |
| Notebook | **LOW-CODE**, preserve original **94 cell count, code/Markdown type sequence, section headings, and order** |
| Primary LLM | `gpt-4o-mini` |
| Evaluation LLM | `gpt-4o` |
| Embeddings | `text-embedding-3-small` |
| Vector store | local/persisted **Chroma** |
| Text splitter | `RecursiveCharacterTextSplitter` |
| Approved baseline | chunk **1200**, overlap **200**, retrieval `k=5`, temperature `0`, top_p `1.0`, max output tokens `650` |
| Retrieval | **source-aware** candidate selection: company/date/type checks and soft diversity, at most k final documents; keep ordinary semantic search as comparator |
| Metric names + pass targets | GPT-4o-based DeepEval **Faithfulness >=0.85**, **Relevance >=0.75**, **Completeness >=0.70**; do not equate scores with ground truth |
| Benchmark | Deterministic seed **42**, **15 development / 5 held-out** goldens; never use five held-out cases or five final broker questions for candidate selection |
| GEPA | Five iterations, Pareto size 3, minibatch size 5, random seed 42, optimizer temperature 0.5, broker answer temperature 0 |
| Three real RAG alternatives | Precision: **600/100, k=3, max_tokens=450**; Balanced: **1200/200, k=5, max_tokens=650**; Context-heavy: **2000/300, k=8, max_tokens=900**; all temp=0/top_p=1 |
| User-facing demo | **ipywidgets inside the original notebook/Google Colab**, not Gradio or a separately hosted app |
| Citations | `[Source N]` in answer and expandable evidence panel with exact source title/type/publisher/ticker/company/date/URL if available, plus verbatim retrieved excerpt |
| Temporal policy | Explicit as-of/reporting dates for historical answers; refuse current-only facts without live data; mixed questions may answer historical part only |
| Human audit | Separate five-final-question sign-off covering unsupported figures, citation support, dates, abstentions, financial advice risk; unresolved issues block business sign-off |
| Answer layout (D15 approved) | Concise broker-first conclusion, dated inline citations, material caveats, expandable evidence detail |
| Paid-call safeguards (D16 approved) | Persist/reuse embedding indexes; small connectivity smoke test in final session only; explicit user confirmation BEFORE any embedding build, evaluation, or GEPA batch; track duration and token usage when returned |
| Submission package (D17 approved) | Executed notebook (after final live run), results-backed PPT (after final run), decision log, evidence/metrics/experiments/human-audit exports, and short run + limitations guide |

Do not invent an API *dollar budget*: the user has not approved one. Confirm paid stages individually at the final session, log known token usage and mark unavailable cost data as unknown. D07 (final winning configuration) remains **provisional until actual results, business audit, and user approval**.

## 3. Required implementation in the notebook

Preserve the existing 94-cell notebook structure *exactly*. Reuse the reference draft where correct, but thoroughly inspect each cell and fix all bugs, misspelled names, undefined references, stale library APIs, logic holes, date/company leakage and Colab incompatibilities. Do not assert the existing draft is fully functional merely because it passes AST parsing.

A. **Installation and environment:** Put install/import/setup code in existing cells, with compatible, reproducible library versions or bounded ranges and a `requirements.txt`. Verify DeepEval/GEPA/Chroma/LangChain APIs by installed package inspection or local tests; if no version is installed, produce a clearly documented runtime compatibility gate. Avoid destructive global upgrades. Accept input data paths from either local VS Code `input/` and Colab `/content/data` or uploaded files. Read `OPENAI_API_KEY` from environment or Colab Secrets; never commit, print or log it. Provide `.env.example` with placeholders and add `.env`, local persistence, cached data, and API response logs to `.gitignore` as appropriate.

B. **Data/metadata:** Load 760 news rows with fields `title,description,content,source,url,published_at,query`; 2,986 price rows `Date,ticker,name,Open,High,Low,Close,Volume`; the Microsoft 10-K with FY2024 document metadata; 20 benchmark rows `question,response,source_hint,supporting_sources,context` ONLY for scoring/optimization. Verify actual row counts and handle Unicode/HTML/XBRL noise carefully. Never confuse price `Date` with news `published_at`. Preserve ticker, period and company identifiers for filtering. Remove empty/noisy documents without silently dropping source coverage. Distinguish a news article's publication date from an SEC fiscal-period end date.

C. **Actual three indexes:** The three configurations must build/reuse distinct Chroma indexes reflecting their respective real SEC chunk sizes/overlaps, not just vary runtime k. Use content hashes of input files plus embedding model and chunk params to prevent stale-index reuse. Do not rebuild for every widget click. Index/document count reporting should avoid leaking huge source text.

D. **Retrieval:** Baseline `k=5`; source-aware candidate retrieval may retrieve more candidates internally but return no more than k final evidence records. Validate source type and company/ticker/date before selecting, avoid forced unrelated filings; expose semantic search comparator. For a multi-source prompt, attempt appropriate news/price/filing support *if available* but don't guarantee all sources. Include a deterministic reason when requested company/period evidence is absent. Supply exact docs actually retrieved to answer generation and to evaluation, with stable unique citation IDs within each answer.

E. **Deterministic policy gate AND final override:** Prioritize predictable compliance: screen for live/current requests, missing filing type/company/period and unsupported events BEFORE generation; permit defensible historical-only partial answers. After model output, enforce a final policy pass to prevent any apparently current quote or invented SEC fact from slipping through. Do not depend solely on instructions in the LLM prompt. Log abstention reasons, evidence gaps, and date limitations. Use a conservative rule for ambiguous live-vs-historical questions (state evidence dates explicitly). Verify these cases with offline synthetic tests.

F. **Answering, UI, and exports:** Build working baseline and optimized answer functions; short broker conclusion with `[Source N]`, caveats and as-of dates. `ipywidgets`: custom question, preset selector, retrieval mode, k, answer temperature, max tokens, submit button, answer view, expandable evidence panel, timing and append-only experiment log. Provide a safe reset/snapshot mechanism and an initial screen explaining the historical dataset. Show clear warning when citation IDs don't refer to retrieved evidence or when the answer lacks support. Don't turn widget clicks into accidental paid batch evaluations.

G. **Benchmark and evaluation:** Seed=42 and freeze a deterministic 15/5 split. Evaluate baseline, GEPA optimized prompt and all three chunking presets; judge with approved DeepEval/GPT-4o metrics. Confirm GEval score direction and threshold semantics for the installed version, including handling missing evidence and abstention separately. Protect against missing/error metrics: do not quietly treat failed calls as zeros or as successes; mark invalid comparisons. Select candidate using development data only, prioritize faithfulness and no unsupported financial claims; report held-out test only AFTER selection is frozen. Benchmark references used for metrics cannot leak into answer context. Retain raw per-example metrics, aggregate metrics, failures, latencies and comments.

H. **GEPA:** Implement a real five-iteration GEPA optimization using compatible package APIs; don't fake success or optimize prompts on held-out questions. Capture optimized prompt, configuration and explanation. Do not run optimization offline without user consent; offline QA may validate class/API imports and construction with mocks only. If the published GEPA API has changed, fix it with version-compatible code and document the exact change.

I. **Five required business questions:** Keep the original five test questions about (1) Apple/Amazon/Alphabet 10-Qs, (2) Amazon news and price sentiment this week, (3) buying Apple and an unverified CEO rumor, (4) buying Bitcoin/current headline, (5) AI-investment comparisons. The final output must explicitly identify unsupported live market claims, absent issuer filings and incomplete comparisons. Capture generated answer, exact retrieved snippets, cite audit, latency, model/settings, and abstention reason per question. Create `five_final_broker_questions.csv` and `five_question_human_signoff.csv`; initially `PENDING`, with pass/revise/fail on each criterion and reviewer notes. Never pre-approve reviews.

J. **Traceability and cost:** Produce CSV/JSON artifacts documenting configurations, dataset snapshot dates, metrics, source citations, audit statuses, offline tests, token/usage fields where actually available, approximate costs ONLY where unit costs have been explicitly verified/configured, and runtimes measured rather than estimated. Incomplete comparisons must be marked incomplete, not sorted as winners.

## 4. PowerPoint — preserve template, don't fabricate results

Starting with the supplied `Project-2 Template.pptx` and using the existing draft where useful, produce `output/UTAIGA_Project_2_Presentation_PreRun.pptx`. Preserve its look/theme and the original assignment sections: Executive Summary; Business Problem; Data Preparation; RAG QA; RAG Evaluation; Tuned RAG. Additional appendix slides are permitted (draft currently has 14 slides). Include the decision/tradeoff table, dataset limitations, source-aware versus semantic retriever, the three chunking experiments, metric thresholds, GEPA settings, temporal abstentions and the human evidence audit. Distinguish **approved design choice**, **expected hypothesis**, **offline-verified outcome**, **requires live results**. Clearly label all pending scores/latency/model answers as `[PENDING LIVE RUN]` (or similarly prominent status). Include neat placeholders/tables that can be filled with actual experimental metrics after the user runs Colab. Never fabricate result charts, model scores or screenshots. Do not change original titles/themes merely to create a fresh-looking presentation.

## 5. Validation path — offline checks plus approved online execution

The user has provided working API access for this session, so you may run the approved online pipeline using the existing credentials and model endpoints when needed. Keep the offline validation checks in place for deterministic QA, but do not block the live run path if the session API is available and the project is ready. Do not use unsupported external paid model paths or unapproved services outside the project’s approved API setup.

You MAY run Python AST/notebook JSON checks, unit tests with fake documents and stubbed LLM/embedding/evaluation calls, CSV schema/data-quality inspections, local slide-file checks and rendering previews, plus the real session-backed model run once the implementation is ready. You MUST NOT run unapproved external or paid model calls outside the configured project environment, solicit new API keys, spin up unapproved Colab sources, or call a live-market feed for unsupported current-data tasks.

Test at minimum:
1. Original vs output notebook cell count=94, exact ordered cell types and original required heading hierarchy.
2. All Python code cells parse (handle `%pip` magic separately) and notebook validates with `nbformat`.
3. Input filenames/schema/record counts and Microsoft-only filing coverage are recognized; benchmark is excluded from retriever.
4. 600/100, 1200/200, 2000/300 resolve to DIFFERENT index identifiers and do not silently reuse mismatched embeddings.
5. Source-aware retrieval enforces company/date/type matching and budget k; comparator remains available.
6. Synthetic current-price, 'this week', stale-event, unsupported 10-Q, historical Microsoft financial, and mixed questions yield appropriate date-aware/abstention handling.
7. Citations resolve to documents retrieved for that particular answer and evidence panel doesn't invent metadata.
8. Split stable at 15/5, no held-out examples in development optimizer/retriever, no final-question tuning.
9. Consent gates prevent paid calls without explicit confirmation; secrets don't leak to output.
10. Human audit sheet defaults to pending; high metric scores cannot override a failed material evidence check.
11. PowerPoint opens, has recognizable original template structure and no fake scores.

Put meaningful tests under `tests/`. Write `output/OFFLINE_QA_REPORT.md` with PASS/FAIL/NOT TESTABLE status and accurate evidence for each check. If a test fails, fix it and rerun. Do not claim end-to-end 'working model' validation based only on syntax checks; live API/Colab behavior remains unverified.

## 6. Files to deliver, and when

**NOW, from VS Code:**
- `output/UTAIGA_Project_2_Low_Code_Final.ipynb` — complete notebook, unchanged 94-cell structure, ready for *later* live execution.
- `output/UTAIGA_Project_2_Presentation_PreRun.pptx` — template-styled draft with clear pending data.
- `output/UTAIGA_Project_2_Decision_Log_FinalDraft.md` — updated decision history and implementation decisions, without pretending test results exist.
- `output/OFFLINE_QA_REPORT.md`, `output/README_FINAL_RUN_AND_SUBMISSION.md` — clear steps to upload data into Colab, set secret, run smoke test, approve paid stages, execute benchmark/GEPA, use widgets, review five answers, generate/collect artifacts, fill PPT, and follow the actual course portal rules once provided.
- `requirements.txt`, `.env.example`, `.gitignore`, offline tests and an optional helper to synchronize observed metrics into PPT placeholders AFTER running (never with fabricated values).

**ONLY LATER, with user approval and live credentials in the final session:** Execute Colab, debug any true runtime/API issues, inspect generated answers, vary controls, review full-source evidence and actual costs, select and approve final configuration, sign off manual audits, update PPT with actual results, and submit files following the real portal instructions. Do not automatically submit or claim submission happened.

## 7. How to work autonomously and report progress

Do not reply with only instructions or another design document: **actually edit the project files**. Start by checking the source content and current package API shape, then implement/repair, run offline tests and prepare final files. Keep the original assets immutable. Work through reasonable engineering details autonomously within approved design; if unavoidable, choose the simpler, more auditable implementation, log it and state the tradeoff. Ask the user only for *material new scope changes*, a real hard dollar spending cap before the first live paid run, and exact submission portal requirements when that stage arrives.

Return a concise completion summary listing produced paths, passing offline validations, untested live integrations, any remaining risks/blockers, and steps for the final Colab session. Document specific implementation decisions in the decision log. Treat missing financial evidence as a first-class expected outcome, not a bug to hide.

## Project Rubric

Treat the rubric below as a mandatory implementation checklist. The notebook, final narrative, and presentation artifacts must explicitly cover each rubric area and the associated observations/comments required for business sign-off.

Criteria
Data Preparation for RAG: load all the provided datasets; choose the text splitting mechanism and parameters with proper rationale; choose the embedding model with proper rationale; set up a vector database/vector store; provide observations/comments.
Points
10

Criteria
Question Answering with RAG: configure an LLM with proper rationale for parameters chosen; define a system prompt for question answering; define the retriever; answer all the test cases; provide comments/observations for all the answers received.
Points
10

Criteria
Evaluating RAG: choose the LLM for evaluation with proper rationale; define evaluation parameters; evaluate responses across all 5 test cases provided; provide comments/observations for the evaluation output.
Points
8

Criteria
Question Answering with Tuned RAG: choose three configurations using combinations of LLM and retrieval parameters (chunk size, chunk overlap, number of chunks to retrieve, LLM temperature, LLM top p, LLM max tokens) with proper rationale; answer all the test cases with each chosen configuration; evaluate all the responses above; provide comments/observations for the evaluation output.
Points
20

Criteria
Actionable Insights and Recommendations: provide key takeaways for the business.
Points
4

Criteria
Presentation/Notebook - Overall Quality: demonstrate structure and flow, crispness, visual appeal, and conclusion and business recommendations; or structure and flow, well-commented code, and conclusion and business recommendations, as applicable to the chosen path.
Points
8