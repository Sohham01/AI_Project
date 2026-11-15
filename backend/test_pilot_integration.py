"""Offline proof of the bounded ingestion-to-review handoff."""

from __future__ import annotations

import unittest

from app.services.ingestion.providers.decodo_html import parse_reddit_html
from app.services.pilot import prepare_end_to_end_pilot


HTML = """
<html><body>
  <shreddit-post id="t3_pilot1" subreddit-prefixed-name="IIMSambalpur"
      post-title="Pilot thread" post-body="Pilot body"
      post-created="2024-03-09T16:00:00Z" score="5" comment-count="2"
      permalink="https://www.reddit.com/r/IIMSambalpur/comments/pilot1/thread/">
    <shreddit-comment id="t1_c1" parent-id="t3_pilot1"
        created-at="2024-03-09T16:00:10Z" score="1">
      <div slot="comment">Top-level question</div>
    </shreddit-comment>
    <shreddit-comment id="t1_c2" parent-id="t1_c1"
        created-at="2024-03-09T16:00:20Z" score="0">
      <div slot="comment">A reply with context</div>
    </shreddit-comment>
  </shreddit-post>
</body></html>
"""


class PilotIntegrationTests(unittest.TestCase):
    def test_html_snapshot_reaches_pipeline_candidates_and_manifest_in_memory(self) -> None:
        posts = parse_reddit_html(HTML)
        result = prepare_end_to_end_pilot(
            posts,
            source_provider="decodo-universal",
            source_mode="direct",
        )

        self.assertEqual(result.summary["status"], "ready_for_review")
        self.assertEqual(result.summary["posts_received"], 1)
        self.assertEqual(result.summary["comments_received"], 2)
        self.assertEqual(result.summary["candidate_rows"], 2)
        self.assertEqual(result.summary["manifest_rows"], 2)
        self.assertFalse(result.summary["writes_performed"])
        self.assertFalse(result.summary["training_performed"])
        self.assertFalse(result.summary["discussion_complete_asserted"])
        self.assertTrue(result.join_report.valid, result.join_report.as_dict())

        self.assertEqual(
            [row["parent_context_status"] for row in result.candidate_rows],
            ["top_level", "available"],
        )
        self.assertTrue(all(row["response_type"] == "" for row in result.candidate_rows))
        self.assertTrue(
            all(record["review"]["status"] == "unreviewed" for record in result.manifest_records)
        )
        self.assertTrue(
            all(record["dataset_approval"]["status"] == "pending" for record in result.manifest_records)
        )
        self.assertEqual(
            [record.document_id for record in result.pipeline_result.records],
            ["comment-c1", "comment-c2"],
        )
        self.assertEqual(
            result.pipeline_result.records[1].created_at.isoformat(),
            "2024-03-09T16:00:20+00:00",
        )
        self.assertEqual(
            result.pipeline_result.records[1].parent_comment_text,
            "Top-level question",
        )
        self.assertEqual(
            result.summary["pipeline_models"]["language"],
            "completed",
        )
        self.assertEqual(
            result.summary["pipeline_models"]["preprocessing"],
            "completed",
        )

    def test_required_source_policy_blocks_without_fabrication(self) -> None:
        posts = parse_reddit_html(HTML)
        result = prepare_end_to_end_pilot(
            posts,
            source_provider="decodo-universal",
            source_mode="direct",
            required_source_fields=("source_record_hash",),
        )
        self.assertEqual(result.summary["status"], "blocked")
        self.assertIn("missing_required_provenance", result.join_report.error_codes)
        self.assertTrue(all(record["source"]["source_record_hash"] is None for record in result.manifest_records))


if __name__ == "__main__":
    unittest.main()
