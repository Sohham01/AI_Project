# Small-Dataset End-to-End Pilot

**Status:** offline preparation complete; live retrieval is still blocked pending
explicit approval. This plan does not authorize a Decodo request, database write,
annotation export, manifest export, or model training.

## Architecture prepared

The offline path is now explicit:

```text
Decodo Universal HTML
  -> decodo_html.parse_reddit_html()
  -> PostData / CommentData
  -> services.pilot.prepare_end_to_end_pilot()
  -> aware downstream records + language/preprocessing
  -> response candidate rows (in memory)
  -> review manifest records (in memory)
  -> candidate/manifest integrity validation
  -> structured pipeline summary
```

The existing `PostData` and `CommentData` schemas remain unchanged. Provider
timestamps are naive UTC; the pilot bridge explicitly attaches UTC at the
downstream boundary rather than silently treating a timestamp as local time.
The downstream pipeline receives `discussion_complete=False` because a capped
HTML page is not proof of a complete thread.

For a supplied static export, structural validation is separate from approval.
Use `load_static_export()` first, then pass the result through
`prepare_authorized_static_pilot()` only after the metadata has
`authorization_status="approved"` and the responsible operator explicitly
confirms `approval_confirmed=True`. `usable_for_pilot` alone is not permission
to process genuine source records. This application gate does not establish
legal rights or replace the supplier's authorization.

## Exact first live request proposal

After explicit approval, the first request should be one request only:

```json
{
  "url": "https://www.reddit.com/r/IIMSambalpur/",
  "proxy_pool": "premium",
  "headless": "html"
}
```

It should be sent through `DecodoProvider.scrape_source()` with:

```text
max_posts=10
comment_limit=25
timeout=10 seconds (or another explicitly approved bounded value)
```

The request must use the configured Web Data API Bearer key from the local
environment; the key must not be pasted into source, logs, or chat. The response
body should be inspected only through an authorized local review step. The
application should record only safe metadata such as request ordinal, source
URL, proxy/rendering configuration, HTTP status, response size if permitted,
provider request ID if supplied, parser diagnostics, and dashboard-reported
usage. It must not log the raw HTML or credentials.

### Credit implication

The first request has a planned maximum of **one Decodo `/scrape` request**.
The exact billed unit depends on the current Decodo account, product, proxy
pool, response, and billing rules. Premium proxy usage and any rendering mode
that causes JavaScript work can consume additional credits; do not infer a free
failure or a fixed price from this repository. Check the account dashboard and
quota immediately before approval.

## Bounded collection plan after the first response

Only if the first HTML response has the expected fields and the operator
approves expansion:

| Limit | Proposed bound |
| --- | ---: |
| Source pages | 2 maximum (`IIMSambalpur`, then optionally `IIMSambalpurUG`) |
| Universal requests | 2 maximum total; no automatic retries |
| Parsed posts | 20 maximum overall, 10 per source page |
| Comments retained | 25 maximum per retained post |
| Post-page follow-ups | 0 in this milestone; separate approval required |
| Pagination/load-more expansion | 0 automatic; diagnostics only |
| Database writes | 0 during the first review pass |
| Candidate/manifest files | 0 during this preparation milestone |

The first source page is the representative inspectable sample. A short page,
missing fields, omitted parents, or a load-more marker must be reported as
partial rather than padded or declared complete. No additional request should
be made merely because the page returns fewer than 10 posts.

## Dataset and provenance record

The future batch should receive an explicit version such as
`pilot-YYYYMMDD-v1` only when the human owner approves the batch. The associated
safe run record should include:

- approved source URL(s) and source community;
- source mode (`direct` for the institution community);
- provider name (`decodo-universal`), proxy pool, and rendering mode;
- request count, per-request status, bounded limits, and parser diagnostics;
- returned/retained post and comment counts;
- duplicate, rejected, missing-parent, and partial-thread counts;
- provider/dashboard usage information when available;
- hashes or snapshot references supplied by the authorized retention process;
- the candidate CSV hash and review-manifest hash after a later approved export.

No source hash, reviewer identity, approval state, or completeness claim may be
fabricated when the provider or retention process does not supply it.

## Partial failure policy

Each request is represented independently. A failed or malformed response is a
structured failure with a safe category/status and no raw response body. Valid
records from a different approved request may remain available for inspection,
but the overall run is `partial_failure` and is not silently released as a
complete dataset. No synthetic post/comment is created to fill a cap.

## Acceptance criteria

The first pilot is acceptable only when all applicable checks pass:

1. The request used the approved Universal body and stayed within the request
   and record caps.
2. The returned HTML was parsed into existing normalized schemas without
   fabricated IDs, timestamps, scores, URLs, parents, or provenance.
3. Duplicate and contradictory identities were handled fail-closed.
4. Partial-thread indicators and omitted parents are visible in diagnostics.
5. Naive ingestion timestamps were explicitly bridged to aware UTC downstream.
6. Language and preprocessing stages run over the bounded in-memory records.
7. Model stages report `unavailable` unless approved, trusted inference models
   are explicitly injected; no model is trained or loaded implicitly.
8. Candidate rows remain blank-label review rows.
9. The manifest is separate, starts unreviewed/pending, and has a valid
   one-to-one `document_id` join with no source conflicts.
10. No file, database, training-artifact, or annotation-area write occurred
    before a separate approval for that operation.

The offline integration test in `backend/test_pilot_integration.py` exercises
these handoffs with synthetic HTML only. It is not evidence of live provider
availability, account quota, billing, response shape beyond the fixture, model
quality, database persistence, or complete Reddit coverage.

The static-export tests also use fabricated records only. They cover structural
validation, duplicate and parent integrity, the explicit authorization gate,
and bounded provenance in the in-memory summary. No genuine Reddit export has
been processed.

The HTML selectors are deliberately conservative and based on explicit
attributes/slots in the offline contract fixtures. The first authorized response
may expose different field names or nesting; any parser adjustment must be made
after inspecting that response and must add a regression fixture before another
live request is considered.
