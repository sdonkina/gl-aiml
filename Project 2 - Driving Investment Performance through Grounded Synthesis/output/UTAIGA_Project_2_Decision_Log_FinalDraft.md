# GlobalEdge Brokerage — Decision Log (Final Draft)

## Scope
This decision log captures the implementation decisions, constraints, and offline validation status for the approved low-code notebook path without claiming any live model results.

## Approved decisions preserved
- Development in VS Code; final live topic deferred to Colab after approval.
- Low-code notebook preserved at 94 cells and section order.
- Historical evidence-first design with explicit abstention on unsupported current market claims.
- Local Chroma for vector storage; no paid API calls in this offline pass.
- DeepEval/GEPA only documented as live-run stages pending approval.

## Data and corpus decisions
- The real dataset names in the workspace are trusted over the stale placeholders in the original notebook prose.
- Only the local financial corpus is used as evidence; benchmark data remains separate from retrieval.
- The SEC corpus is Microsoft FY2024 10-K only; no fabricated filing coverage is allowed.

## Notebook implementation decisions
- The low-code notebook structure was kept intact.
- Placeholder names were corrected to the actual dataset files in the workspace.
- Validation logic was built to check the real notebook shape and guard against hidden format drift.

## Offline validation status
- Notebook structure validation passes.
- Dataset existence and names are verified.
- Source-aware retrieval guardrails are tested offline.
- No live API calls or paid model execution were performed.

## Pending live-run stage
The following remain intentionally marked as pending until the user confirms the final execution session:
- live embedding generation
- benchmark scoring
- GEPA optimization
- final answer generation by model
- actual metrics and latency measurement
- PowerPoint finalization with real results and business sign-off

## Risk notice
The project is designed to answer historical financial research questions conservatively and abstain when evidence is missing. Claims about current market conditions or unsupported investment advice remain blocked by design.
