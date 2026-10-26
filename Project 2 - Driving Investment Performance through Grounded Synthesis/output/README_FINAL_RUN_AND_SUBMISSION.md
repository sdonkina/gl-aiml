# Final Run and Submission Guide

## Purpose
This guide explains the final live execution stage for the GlobalEdge Brokerage financial RAG project. All steps below are intentionally deferred until the user approves the final live-run session and provides credentials or access.

## 1. Prepare the environment
1. Open the notebook in Google Colab.
2. Upload or mount the project dataset folder.
3. Set the required secret(s) in Colab Secrets if the user approves live model access.
4. Only then install the required runtime dependencies from `requirements.txt`.

## 2. Run the notebook
1. Open the final notebook in Colab.
2. Run all cells in order.
3. Review the widget-driven question interface.
4. Confirm the historical-data guardrails are active before any answer generation.

## 3. Approved live stages
Only after explicit confirmation should the user run:
- embedding/index creation
- DeepEval benchmark scoring
- GEPA optimization
- final answer generation against live model endpoints

## 4. Human audit signoff
After the final live run, review the five broker questions and confirm all of the following:
- no unsupported financial claims
- all citations correspond to retrieved evidence
- as-of dates are explicit
- historical vs live questions are labeled correctly
- no business recommendation is made without evidence

## 5. Submission package
Once the live run is approved and results are reviewed, collect and submit:
- final notebook
- final PPT with actual metrics inserted
- decision log
- benchmark exports and audit logs
- final run summary and note any blockers

## 6. Offline status
The current project state is offline-validated only. No live run, OpenAI call, or paid model use has been performed in this workspace.
