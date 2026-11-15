"""
Comprehensive tests for the topic discovery stage (TF-IDF -> NMF).

Tests cover:
- Empty corpus and too-small corpus handling
- Determinism (same input -> same topics, twice)
- Configurable n_topics and n_keywords
- Topic keyword structure and label derivation
- Document-to-topic mapping and traceability metadata
- Duplicate document IDs
- Multilingual / code-mixed corpus handling and domain-term preservation
- Input text is never modified
- Model metadata, version and configuration snapshot
- Invalid configuration rejection
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.nlp.topic_discovery import (
    DocumentTopicAssignment,
    NMFTopicDiscoverer,
    NMFConfig,
    TFIDFConfig,
    TopicDiscovery,
    TopicDiscoveryResult,
    TopicDocument,
    TopicInfo,
)

from app.services.nlp.preprocessing import MultilingualTextPreprocessor


# A small, representative corpus in the languages the project targets.
# Deliberately themed so a working model has real structure to find.
SAMPLES = [
    # --- placements / PPO ---
    ("p01", "IIM Sambalpur placement season was very competitive and the PPO process was smooth for most students", "en", "latin", False, ["en"]),
    ("p02", "bhai placement ka kya scene hai PPO mila ya nhi mila confusion", "hi", "latin", True, ["hi", "en"]),
    ("p03", "campus placements PPO selection process is quite unpredictable every single year", "en", "latin", False, ["en"]),
    ("p04", "placement aur PPO dono ke liye interview rounds bahut tough the", "hi", "latin", True, ["hi", "en"]),
    # --- CAT exam ---
    ("p05", "CAT exam preparation needs serious mock tests and good coaching for the MBA admission cycle", "en", "latin", False, ["en"]),
    ("p06", "CAT exam di taiyari aur coaching MBA admission ke liye kitni zaroori hai", "hi", "latin", True, ["hi", "en"]),
    ("p07", "mock tests for the CAT exam are the best way to prepare for MBA admission", "en", "latin", False, ["en"]),
    # --- hostel / mess ---
    ("p08", "hostel mess food quality is really bad and the mess charges are too high every month", "en", "latin", False, ["en"]),
    ("p09", "hostel ka mess aur food quality kharab hai mess charges bahut zyada hai", "hi", "latin", True, ["hi", "en"]),
    ("p10", "mess food quality hostel charges should be reduced by the administration", "en", "latin", False, ["en"]),
    # --- fees / scholarship / ROI ---
    ("p11", "the fees are very high and the ROI is not worth it compared to the MBA return", "en", "latin", False, ["en"]),
    ("p12", "fees scholarship aur ROI ka kya scene hai MBA ka return justify karta hai ya nhi", "hi", "latin", True, ["hi", "en"]),
    ("p13", "scholarship fees and ROI should be discussed openly with the students", "en", "latin", False, ["en"]),
    # --- internships ---
    ("p14", "internship opportunities during the summer are limited but the internship projects are good", "en", "latin", False, ["en"]),
    ("p15", "summer internship ke liye company kaise select kare internship projects mast hote hai", "hi", "latin", True, ["hi", "en"]),
    ("p16", "internship MBA ke saath karna chahiye internship experience bahut important hai", "hi", "latin", True, ["hi", "en"]),
    # --- native script ---
    ("p17", "आईआईएम संबलपुर में प्लेसमेंट बहुत अच्छा है", "hi", "devanagari", False, ["hi"]),
    ("p18", "हॉस्टल का मेस खाना अच्छा नहीं है", "hi", "devanagari", False, ["hi"]),
    ("p19", "ক্যাট পরীক্ষার প্রস্তুতি খুব কঠিন", "bn", "bengali", False, ["bn"]),
    ("p20", "প্লেসমেন্ট প্রক্রিয়া ভালো হয়েছে", "bn", "bengali", False, ["bn"]),
    ("p21", "ହସ୍ଟେଲ ଖାବା ଭଲ ନାହିଁ", "or", "odia", False, ["or"]),
    ("p22", "ପ୍ଲେସମେଣ୍ଟ ବହୁତ ଭଲ ହେଉଛି", "or", "odia", False, ["or"]),
    # --- romanized + emoji ---
    ("p23", "bhaiiii \U0001f62d placement ka kya scene hai??? \U0001f525", "hi", "latin", True, ["hi", "en"]),
    ("p24", "CAT exam pe mood off \U0001f625 coaching se thoda relief mila", "hi", "latin", True, ["hi", "en"]),
]


def build_corpus() -> list:
    """Build the sample corpus as TopicDocument objects."""
    return [
        TopicDocument(
            document_id=doc_id,
            text=text,
            post_id=doc_id if doc_id.startswith("p") else None,
            original_text=text,
            language=language,
            script=script,
            is_code_mixed=is_code_mixed,
            detected_languages=detected,
        )
        for doc_id, text, language, script, is_code_mixed, detected in SAMPLES
    ]


class TestTopicDiscoveryInterface(unittest.TestCase):
    """The interface contract that a future embedding implementation must honour."""

    def test_is_abstract(self):
        with self.assertRaises(TypeError):
            TopicDiscovery()

    def test_nmf_implements_interface(self):
        self.assertIsInstance(NMFTopicDiscoverer(), TopicDiscovery)

    def test_result_types_exported(self):
        self.assertTrue(issubclass(TopicDiscoveryResult, object))
        for model in (TopicInfo, DocumentTopicAssignment, TopicDocument):
            self.assertTrue(hasattr(model, "model_fields"))

    def test_nmf_topic_discovery_uses_own_name(self):
        discoverer = NMFTopicDiscoverer()
        self.assertEqual(discoverer.name, "nmf")
        self.assertEqual(discoverer.version, "v1")


class TestEmptyAndSmallCorpus(unittest.TestCase):
    def test_empty_corpus(self):
        result = NMFTopicDiscoverer().discover([])
        self.assertEqual(result.n_documents, 0)
        self.assertEqual(result.topics, [])
        self.assertEqual(result.documents, [])
        self.assertEqual(result.vocabulary_size, 0)
        self.assertTrue(any("empty_corpus" in w for w in result.warnings))

    def test_empty_corpus_still_has_metadata(self):
        result = NMFTopicDiscoverer().discover([])
        self.assertEqual(result.topic_model, "nmf")
        self.assertEqual(result.topic_model_version, "v1")

    def test_none_corpus(self):
        result = NMFTopicDiscoverer().discover(None)
        self.assertEqual(result.n_documents, 0)

    def test_too_small_corpus_raises(self):
        docs = build_corpus()[:2]
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(n_topics=8).discover(docs)

    def test_too_small_corpus_error_message_is_actionable(self):
        docs = build_corpus()[:2]
        with self.assertRaises(ValueError) as ctx:
            NMFTopicDiscoverer(n_topics=5).discover(docs)
        self.assertIn("too-small corpus", str(ctx.exception))

    def test_corpus_without_usable_tokens_raises(self):
        docs = [TopicDocument(document_id="a", text="   "), TopicDocument(document_id="b", text="")]
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(n_topics=1).discover(docs)

    def test_n_topics_equal_to_document_count_is_allowed(self):
        docs = build_corpus()[:3]
        result = NMFTopicDiscoverer(n_topics=3).discover(docs)
        self.assertEqual(len(result.topics), 3)


class TestDeterminism(unittest.TestCase):
    def test_repeated_runs_identical(self):
        discoverer = NMFTopicDiscoverer(n_topics=6)
        r1 = discoverer.discover(build_corpus())
        r2 = discoverer.discover(build_corpus())
        self.assertEqual(
            [t.keywords for t in r1.topics], [t.keywords for t in r2.topics]
        )
        self.assertEqual(
            [t.keyword_weights for t in r1.topics],
            [t.keyword_weights for t in r2.topics],
        )

    def test_assignments_identical(self):
        discoverer = NMFTopicDiscoverer(n_topics=6)
        r1 = discoverer.discover(build_corpus())
        r2 = discoverer.discover(build_corpus())
        self.assertEqual(
            [d.topic_distribution for d in r1.documents],
            [d.topic_distribution for d in r2.documents],
        )
        self.assertEqual(
            [d.dominant_topic_id for d in r1.documents],
            [d.dominant_topic_id for d in r2.documents],
        )

    def test_two_instances_agree(self):
        a = NMFTopicDiscoverer(n_topics=6).discover(build_corpus())
        b = NMFTopicDiscoverer(n_topics=6).discover(build_corpus())
        self.assertEqual([t.keywords for t in a.topics], [t.keywords for t in b.topics])

    def test_default_random_state_is_pinned(self):
        self.assertEqual(NMFConfig().random_state, 42)
        self.assertEqual(NMFTopicDiscoverer().config_dict()["nmf"]["random_state"], 42)

    def test_timestamp_excluded_by_default(self):
        result = NMFTopicDiscoverer(n_topics=6).discover(build_corpus())
        self.assertIsNone(result.discovered_at)

    def test_timestamp_opt_in(self):
        result = NMFTopicDiscoverer(n_topics=6, include_timestamp=True).discover(
            build_corpus()
        )
        self.assertIsNotNone(result.discovered_at)
        # ISO-8601 with timezone
        self.assertIn("+00:00", result.discovered_at)


class TestConfigurability(unittest.TestCase):
    def test_default_n_topics_is_eight(self):
        self.assertEqual(NMFTopicDiscoverer().n_topics, 8)
        self.assertEqual(NMFConfig().n_topics, 8)

    def test_n_topics_configurable_via_constructor(self):
        result = NMFTopicDiscoverer(n_topics=3).discover(build_corpus())
        self.assertEqual(result.n_topics, 3)
        self.assertEqual(len(result.topics), 3)

    def test_n_topics_configurable_via_call(self):
        result = NMFTopicDiscoverer(n_topics=8).discover(build_corpus(), n_topics=4)
        self.assertEqual(result.n_topics, 4)
        self.assertEqual(len(result.topics), 4)
        self.assertEqual(len(result.documents[0].topic_distribution), 4)

    def test_n_topics_via_nmf_config(self):
        result = NMFTopicDiscoverer(nmf_config=NMFConfig(n_topics=2)).discover(build_corpus())
        self.assertEqual(len(result.topics), 2)

    def test_n_keywords_configurable(self):
        result = NMFTopicDiscoverer(n_topics=4, n_keywords=3).discover(build_corpus())
        for topic in result.topics:
            self.assertEqual(len(topic.keywords), 3)

    def test_tfidf_configurable(self):
        cfg = TFIDFConfig(
            max_features=500,
            min_df=2,
            max_df=1.0,
            ngram_range=(1, 1),
            stop_words="english",
        )
        result = NMFTopicDiscoverer(n_topics=3, tfidf_config=cfg).discover(build_corpus())
        self.assertLessEqual(result.vocabulary_size, 500)
        self.assertEqual(result.config["tfidf"]["ngram_range"], [1, 1])
        self.assertEqual(result.config["tfidf"]["stop_words"], "english")
        self.assertEqual(result.config["tfidf"]["min_df"], 2)

    def test_unigrams_only_drops_bigrams(self):
        uni = NMFTopicDiscoverer(
            n_topics=3, tfidf_config=TFIDFConfig(ngram_range=(1, 1))
        ).discover(build_corpus())
        for topic in uni.topics:
            self.assertTrue(all(" " not in kw for kw in topic.keywords))

    def test_bigrams_appear_by_default(self):
        result = NMFTopicDiscoverer(n_topics=4).discover(build_corpus())
        keywords = [kw for t in result.topics for kw in t.keywords]
        self.assertTrue(any(" " in kw for kw in keywords))

    def test_custom_stop_words_accepted(self):
        cfg = TFIDFConfig(stop_words=["placement", "campus"])
        result = NMFTopicDiscoverer(n_topics=3, tfidf_config=cfg).discover(build_corpus())
        keywords = {kw for t in result.topics for kw in t.keywords}
        self.assertNotIn("placement", keywords)

    def test_nmf_config_snapshotted(self):
        cfg = NMFConfig(n_topics=3, max_iter=77, init="nndsvda")
        result = NMFTopicDiscoverer(nmf_config=cfg).discover(build_corpus())
        self.assertEqual(result.config["nmf"]["max_iter"], 77)
        self.assertEqual(result.config["n_topics"], 3)


class TestTopicKeywordStructure(unittest.TestCase):
    def setUp(self):
        self.result = NMFTopicDiscoverer(n_topics=6, n_keywords=8).discover(build_corpus())

    def test_topic_ids_are_sequential(self):
        self.assertEqual([t.topic_id for t in self.result.topics], list(range(6)))

    def test_labels_are_generic_and_model_derived(self):
        for topic in self.result.topics:
            self.assertEqual(topic.label, f"Topic {topic.topic_id}")
            # No hard-coded IIM category names anywhere in the labels
            self.assertNotIn("placement", topic.label.lower())

    def test_keyword_counts(self):
        for topic in self.result.topics:
            self.assertEqual(len(topic.keywords), 8)
            self.assertEqual(len(topic.keyword_weights), 8)

    def test_keywords_are_unique_non_empty_strings(self):
        for topic in self.result.topics:
            self.assertEqual(len(set(topic.keywords)), len(topic.keywords))
            for kw in topic.keywords:
                self.assertIsInstance(kw, str)
                self.assertTrue(kw.strip())

    def test_keyword_weights_descend(self):
        for topic in self.result.topics:
            weights = topic.keyword_weights
            self.assertEqual(weights, sorted(weights, reverse=True))
            self.assertTrue(all(w >= 0 for w in weights))

    def test_topics_have_distinct_keywords(self):
        seen = [set(t.keywords) for t in self.result.topics]
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                self.assertNotEqual(seen[i], seen[j])

    def test_document_counts_sum_to_corpus_size(self):
        total = sum(t.document_count for t in self.result.topics)
        self.assertEqual(total, self.result.n_documents)

    def test_document_ratios_sum_to_one(self):
        total = sum(t.document_ratio for t in self.result.topics)
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_keyword_count_never_exceeds_vocabulary(self):
        for topic in self.result.topics:
            self.assertLessEqual(len(topic.keywords), self.result.vocabulary_size)

    def test_topic_by_id_helper(self):
        self.assertIsNotNone(self.result.topic_by_id(0))
        self.assertIsNone(self.result.topic_by_id(999))

    def test_known_themes_are_discovered(self):
        """The corpus is thematically structured, so topics must be separable."""
        result = NMFTopicDiscoverer(n_topics=6, n_keywords=15).discover(build_corpus())
        keywords = {kw for t in result.topics for kw in t.keywords}
        for theme_term in ("placement", "cat", "hostel", "mess", "internship"):
            self.assertIn(theme_term, keywords)

    def test_coherent_themes_share_a_dominant_topic(self):
        """Documents about the same subject should land on the same topic."""
        themes = {
            "cat": ["p05", "p06", "p07"],
            "hostel": ["p08", "p09", "p10"],
            "internship": ["p14", "p15", "p16"],
        }
        for theme, doc_ids in themes.items():
            assigned = [
                d.dominant_topic_id
                for d in self.result.documents
                if d.document_id in doc_ids
            ]
            self.assertEqual(len(assigned), len(doc_ids))
            self.assertEqual(
                len(set(assigned)), 1, f"{theme} docs scattered across {set(assigned)}"
            )

    def test_configurable_stopwords_improve_topic_separation(self):
        """
        Configurability that matters: an explicitly supplied neutral list of
        English + Romanized-Indic function words removes the 'ka / kya / scene'
        filler topic that dominates the conservative default.  The default
        itself stays conservative (no language-specific removal) — this proves
        the knob works without changing the default.
        """
        filler_stopwords = [
            "hai", "ka", "ki", "ke", "kya", "scene", "bhai", "aur", "ya",
            "nahi", "bahut", "the", "is", "a", "and", "of", "for", "to",
            "in", "are", "with", "se", "me", "mein", "par", "ho", "liye",
        ]
        result = NMFTopicDiscoverer(
            n_topics=6, n_keywords=10, tfidf_config=TFIDFConfig(stop_words=filler_stopwords)
        ).discover(build_corpus())
        keywords = {kw for t in result.topics for kw in t.keywords}
        self.assertNotIn("kya scene", keywords)
        self.assertNotIn("scene hai", keywords)
        self.assertIn("placement", keywords)
        self.assertIn("ppo", keywords)

    def test_default_config_applies_no_stopword_removal(self):
        """Requirement: no language-specific stop-word removal by default."""
        result = NMFTopicDiscoverer(n_topics=6, n_keywords=10).discover(build_corpus())
        self.assertIsNone(result.config["tfidf"]["stop_words"])
        keywords = {kw for t in result.topics for kw in t.keywords}
        self.assertTrue({"hai", "ka", "kya", "scene"} & keywords)


class TestDocumentTopicMapping(unittest.TestCase):
    def setUp(self):
        self.docs = build_corpus()
        self.result = NMFTopicDiscoverer(n_topics=6).discover(self.docs)

    def test_one_assignment_per_document(self):
        self.assertEqual(len(self.result.documents), len(self.docs))

    def test_assignment_order_matches_input(self):
        self.assertEqual(
            [d.document_id for d in self.result.documents],
            [d.document_id for d in self.docs],
        )

    def test_distribution_has_one_entry_per_topic(self):
        for assignment in self.result.documents:
            self.assertEqual(len(assignment.topic_distribution), self.result.n_topics)

    def test_distribution_sums_to_one(self):
        for assignment in self.result.documents:
            total = sum(assignment.topic_distribution.values())
            self.assertAlmostEqual(total, 1.0, places=3)

    def test_distribution_weights_are_probabilities(self):
        for assignment in self.result.documents:
            for weight in assignment.topic_distribution.values():
                self.assertGreaterEqual(weight, 0.0)
                self.assertLessEqual(weight, 1.0)

    def test_dominant_topic_matches_argmax(self):
        for assignment in self.result.documents:
            best = max(
                assignment.topic_distribution.items(), key=lambda kv: (kv[1], -kv[0])
            )
            self.assertEqual(assignment.dominant_topic_id, best[0])
            self.assertAlmostEqual(
                assignment.dominant_topic_score, best[1], places=3
            )

    def test_dominant_label_consistent_with_topic_table(self):
        for assignment in self.result.documents:
            topic = self.result.topic_by_id(assignment.dominant_topic_id)
            self.assertEqual(assignment.dominant_topic_label, topic.label)

    def test_counts_consistent_with_dominant_assignments(self):
        counts = {}
        for assignment in self.result.documents:
            counts[assignment.dominant_topic_id] = (
                counts.get(assignment.dominant_topic_id, 0) + 1
            )
        for topic in self.result.topics:
            self.assertEqual(topic.document_count, counts.get(topic.topic_id, 0))

    def test_metadata_round_trips(self):
        for source, assignment in zip(self.docs, self.result.documents):
            self.assertEqual(assignment.language, source.language)
            self.assertEqual(assignment.script, source.script)
            self.assertEqual(assignment.is_code_mixed, source.is_code_mixed)
            self.assertEqual(assignment.detected_languages, source.detected_languages)
            self.assertEqual(assignment.post_id, source.post_id)
            self.assertEqual(assignment.comment_id, source.comment_id)
            self.assertEqual(assignment.original_text, source.original_text)

    def test_comment_documents_are_traceable(self):
        docs = [
            TopicDocument(document_id="c1", comment_id="cmt-1", text="hostel mess food is bad"),
            TopicDocument(document_id="c2", comment_id="cmt-2", text="hostel mess charges are high"),
            TopicDocument(document_id="c3", comment_id="cmt-3", text="CAT exam preparation coaching"),
        ]
        result = NMFTopicDiscoverer(n_topics=2).discover(docs)
        self.assertEqual(
            [d.comment_id for d in result.documents], ["cmt-1", "cmt-2", "cmt-3"]
        )

    def test_get_document_helper(self):
        assignment = self.result.get_document("p01")
        self.assertIsNotNone(assignment)
        self.assertEqual(assignment.document_id, "p01")
        self.assertIsNone(self.result.get_document("does-not-exist"))


class TestDuplicateDocumentIds(unittest.TestCase):
    def test_duplicates_are_all_assigned(self):
        docs = [
            TopicDocument(document_id="dup", text="placement PPO process campus season"),
            TopicDocument(document_id="dup", text="placement PPO process campus season"),
            TopicDocument(document_id="other", text="CAT exam preparation coaching MBA"),
        ]
        result = NMFTopicDiscoverer(n_topics=2).discover(docs)
        self.assertEqual(len(result.documents), 3)
        self.assertEqual(result.n_documents, 3)

    def test_duplicates_are_reported_as_warning(self):
        docs = [
            TopicDocument(document_id="dup", text="hostel mess food quality charges"),
            TopicDocument(document_id="dup", text="hostel mess food quality charges"),
        ]
        result = NMFTopicDiscoverer(n_topics=1).discover(docs)
        self.assertTrue(any("duplicate_document_ids" in w for w in result.warnings))

    def test_no_warning_when_ids_unique(self):
        result = NMFTopicDiscoverer(n_topics=3).discover(build_corpus())
        self.assertFalse(any("duplicate_document_ids" in w for w in result.warnings))

    def test_duplicate_documents_get_consistent_topics(self):
        docs = [
            TopicDocument(document_id="dup", text="CAT exam preparation coaching mock tests"),
            TopicDocument(document_id="dup", text="CAT exam preparation coaching mock tests"),
        ]
        result = NMFTopicDiscoverer(n_topics=1).discover(docs)
        first, second = result.documents
        self.assertEqual(first.dominant_topic_id, second.dominant_topic_id)
        self.assertEqual(first.topic_distribution, second.topic_distribution)


class TestMultilingualAndCodeMixed(unittest.TestCase):
    def setUp(self):
        self.docs = build_corpus()
        self.result = NMFTopicDiscoverer(n_topics=6).discover(self.docs)

    def test_non_latin_scripts_reach_the_vocabulary(self):
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 1), token_pattern=TFIDFConfig().token_pattern
        )
        vectorizer.fit([d.text for d in self.docs])
        vocab = set(vectorizer.vocabulary_.keys())
        devanagari = [t for t in vocab if re.search(r"[\u0900-\u097f]", t)]
        bengali = [t for t in vocab if re.search(r"[\u0980-\u09ff]", t)]
        odia = [t for t in vocab if re.search(r"[\u0b00-\u0b7f]", t)]
        self.assertTrue(devanagari, "no Devanagari terms in vocabulary")
        self.assertTrue(bengali, "no Bengali terms in vocabulary")
        self.assertTrue(odia, "no Odia terms in vocabulary")

    def test_code_mixed_documents_are_assigned(self):
        mixed = [d for d in self.result.documents if d.is_code_mixed]
        self.assertGreaterEqual(len(mixed), 5)
        for assignment in mixed:
            self.assertGreaterEqual(len(assignment.detected_languages), 1)
            self.assertIn(assignment.dominant_topic_id, range(self.result.n_topics))

    def test_code_mixed_flag_is_preserved(self):
        for source, assignment in zip(self.docs, self.result.documents):
            self.assertEqual(assignment.is_code_mixed, source.is_code_mixed)

    def test_domain_terms_are_preserved_by_default_config(self):
        """IIM/CAT/PPO/ROI/MBA and campus terms must survive tokenisation."""
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 1), token_pattern=TFIDFConfig().token_pattern
        )
        vectorizer.fit([d.text for d in self.docs])
        vocab = set(vectorizer.vocabulary_.keys())
        for term in ("iim", "cat", "ppo", "roi", "mba", "placement", "internship", "hostel", "mess"):
            self.assertIn(term, vocab, f"domain term '{term}' was dropped")

    def test_no_language_specific_stopword_removal_by_default(self):
        """Romanized function words carry the code-switch structure."""
        result = NMFTopicDiscoverer(
            n_topics=2, tfidf_config=TFIDFConfig(stop_words=None)
        ).discover(
            [
                TopicDocument(document_id="m1", text="placement ka kya scene hai bhai"),
                TopicDocument(document_id="m2", text="placement ka scene kya hai bhai"),
                TopicDocument(document_id="m3", text="placement scene bhai ka kya hai"),
            ]
        )
        keywords = {kw for t in result.topics for kw in t.keywords}
        self.assertTrue({"hai", "ka", "kya", "scene"} & keywords)

    def test_text_is_not_translated(self):
        """Native-script documents must not be converted to English."""
        devanagari_doc = next(d for d in self.result.documents if d.script == "devanagari")
        self.assertRegex(devanagari_doc.text, r"[\u0900-\u097f]")

    def test_emoji_placeholders_excluded_from_features(self):
        result = NMFTopicDiscoverer(n_topics=3).discover(
            [
                TopicDocument(document_id="e1", text="bhaiiii <EMOJI_loudly_crying_face> placement ka scene hai"),
                TopicDocument(document_id="e2", text="placement <EMOJI_question_mark> campus PPO process"),
                TopicDocument(document_id="e3", text="campus placement PPO <EMOJI_fire> season"),
            ]
        )
        keywords = {kw for t in result.topics for kw in t.keywords}
        self.assertFalse([kw for kw in keywords if kw.startswith("EMOJI")])
        self.assertTrue(any("placeholders_dropped" in w for w in result.warnings))

    def test_placeholder_drop_can_be_disabled(self):
        docs = [
            TopicDocument(document_id="e1", text="<EMOJI_fire> placement campus season"),
            TopicDocument(document_id="e2", text="<EMOJI_fire> placement PPO process"),
        ]
        result = NMFTopicDiscoverer(
            n_topics=2,
            nmf_config=NMFConfig(drop_placeholder_tokens=False),
        ).discover(docs)
        keywords = {kw for t in result.topics for kw in t.keywords}
        self.assertIn("emoji_fire", keywords)

    def test_url_mentions_do_not_form_a_topic(self):
        docs = build_corpus() + [
            TopicDocument(document_id="u1", text="check <URL> and <MENTION> for details", post_id="u1"),
            TopicDocument(document_id="u2", text="see <URL> <MENTION> link only", post_id="u2"),
        ]
        result = NMFTopicDiscoverer(n_topics=6).discover(docs)
        keywords = {kw for t in result.topics for kw in t.keywords}
        self.assertFalse([kw for kw in keywords if kw in ("url", "mention")])

    def test_preprocessed_pipeline_output_feeds_topic_discovery(self):
        """End-to-end: preprocessing output must be directly usable as input."""
        preprocessor = MultilingualTextPreprocessor()
        raw = [
            "broooo 😭 IIM ka placement kya scene hai??",
            "CAT exam ki taiyari aur coaching kaise kare",
            "hostel mess food quality bahut kharab hai",
        ]
        docs = []
        for i, text in enumerate(raw):
            pre = preprocessor.preprocess(text, language="hi", script="latin", is_code_mixed=True)
            docs.append(
                TopicDocument(
                    document_id=f"e2e{i}",
                    text=pre.normalized_text,
                    original_text=pre.original_text,
                    language="hi",
                    script="latin",
                    is_code_mixed=True,
                    detected_languages=["hi", "en"],
                )
            )
        result = NMFTopicDiscoverer(n_topics=2).discover(docs)
        self.assertEqual(result.n_documents, 3)
        for assignment in result.documents:
            self.assertIn(assignment.dominant_topic_id, (0, 1))


class TestInputNotModified(unittest.TestCase):
    def test_input_documents_unchanged(self):
        docs = build_corpus()
        before = [d.model_dump() for d in docs]
        NMFTopicDiscoverer(n_topics=6).discover(docs)
        after = [d.model_dump() for d in docs]
        self.assertEqual(before, after)

    def test_input_text_unchanged(self):
        docs = build_corpus()
        before = [d.text for d in docs]
        NMFTopicDiscoverer(n_topics=6).discover(docs)
        self.assertEqual(before, [d.text for d in docs])

    def test_original_text_unchanged(self):
        docs = build_corpus()
        before = [d.original_text for d in docs]
        result = NMFTopicDiscoverer(n_topics=6).discover(docs)
        self.assertEqual(before, [d.original_text for d in result.documents])

    def test_assignment_text_is_verbatim_input(self):
        docs = build_corpus()
        result = NMFTopicDiscoverer(n_topics=6).discover(docs)
        for source, assignment in zip(docs, result.documents):
            self.assertEqual(assignment.text, source.text)

    def test_emoji_tokens_remain_in_result_text(self):
        """Preprocessing output must be echoed back untouched, placeholders included."""
        docs = [
            TopicDocument(document_id="a", text="<EMOJI_fire> placement PPO campus"),
            TopicDocument(document_id="b", text="<EMOJI_fire> placement PPO campus"),
        ]
        result = NMFTopicDiscoverer(n_topics=1).discover(docs)
        self.assertEqual(result.documents[0].text, "<EMOJI_fire> placement PPO campus")

    def test_source_corpus_list_not_reordered_or_extended(self):
        docs = build_corpus()
        n = len(docs)
        NMFTopicDiscoverer(n_topics=6).discover(docs)
        self.assertEqual(len(docs), n)
        self.assertEqual(docs[0].document_id, "p01")


class TestModelMetadata(unittest.TestCase):
    def setUp(self):
        self.result = NMFTopicDiscoverer(n_topics=6).discover(build_corpus())

    def test_model_name_and_version(self):
        self.assertEqual(self.result.topic_model, "nmf")
        self.assertEqual(self.result.topic_model_version, "v1")

    def test_metadata_present_even_on_empty_corpus(self):
        result = NMFTopicDiscoverer().discover([])
        self.assertEqual(result.topic_model, "nmf")
        self.assertEqual(result.topic_model_version, "v1")

    def test_config_snapshot_contains_all_knobs(self):
        config = self.result.config
        self.assertIn("n_topics", config)
        self.assertIn("n_keywords", config)
        self.assertIn("tfidf", config)
        self.assertIn("nmf", config)
        for key in (
            "max_features", "min_df", "max_df", "ngram_range", "stop_words",
            "lowercase", "sublinear_tf", "token_pattern", "norm", "use_idf", "smooth_idf",
        ):
            self.assertIn(key, config["tfidf"])
        for key in (
            "init", "solver", "beta_loss", "random_state", "max_iter", "tol",
            "alpha_W", "alpha_H", "l1_ratio", "drop_placeholder_tokens",
        ):
            self.assertIn(key, config["nmf"])

    def test_config_is_json_serializable(self):
        import json
        json.dumps(self.result.config)
        json.dumps(self.result.model_dump())

    def test_library_versions_recorded(self):
        self.assertIn("scikit-learn", self.result.library_versions)
        self.assertIn("numpy", self.result.library_versions)

    def test_corpus_statistics(self):
        self.assertEqual(self.result.n_documents, len(SAMPLES))
        self.assertEqual(self.result.n_topics, 6)
        self.assertGreater(self.result.vocabulary_size, 0)

    def test_result_is_pydantic_model(self):
        self.assertIsInstance(self.result, TopicDiscoveryResult)


class TestInvalidConfiguration(unittest.TestCase):
    def test_zero_n_topics(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(n_topics=0)

    def test_negative_n_topics(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(n_topics=-3)

    def test_non_integer_n_topics(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(n_topics="eight")

    def test_boolean_n_topics(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(n_topics=True)

    def test_zero_n_keywords(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(n_keywords=0)

    def test_invalid_n_topics_at_call_time(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(n_topics=4).discover(build_corpus(), n_topics=0)

    def test_invalid_max_features(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(tfidf_config=TFIDFConfig(max_features=0))

    def test_invalid_min_df_int(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(tfidf_config=TFIDFConfig(min_df=-1))

    def test_invalid_min_df_float_range(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(tfidf_config=TFIDFConfig(min_df=1.5))

    def test_invalid_max_df_float_range(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(tfidf_config=TFIDFConfig(max_df=0.0))

    def test_invalid_ngram_range(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(tfidf_config=TFIDFConfig(ngram_range=(2, 1)))

    def test_invalid_ngram_range_zero(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(tfidf_config=TFIDFConfig(ngram_range=(0, 1)))

    def test_invalid_norm(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(tfidf_config=TFIDFConfig(norm="l3"))

    def test_invalid_token_pattern(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(tfidf_config=TFIDFConfig(token_pattern="(?u)\\w+("))

    def test_language_specific_stopword_list_rejected(self):
        """Hindi/Bengali/Odia stop-word removal is not supported by design."""
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(tfidf_config=TFIDFConfig(stop_words="hindi"))

    def test_stop_words_wrong_type(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(tfidf_config=TFIDFConfig(stop_words=42))

    def test_stop_words_list_with_non_strings(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(tfidf_config=TFIDFConfig(stop_words=["ok", 7]))

    def test_invalid_max_iter(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(nmf_config=NMFConfig(max_iter=0))

    def test_invalid_tol(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(nmf_config=NMFConfig(tol=-1.0))

    def test_invalid_init(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(nmf_config=NMFConfig(init="magic"))

    def test_invalid_beta_loss(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(nmf_config=NMFConfig(beta_loss="kullback"))

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(nmf_config=NMFConfig(alpha_W=-1.0))

    def test_invalid_l1_ratio(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(nmf_config=NMFConfig(l1_ratio=1.5))

    def test_invalid_n_keywords_in_config(self):
        with self.assertRaises(ValueError):
            NMFTopicDiscoverer(nmf_config=NMFConfig(n_keywords=0))


if __name__ == "__main__":
    unittest.main()
