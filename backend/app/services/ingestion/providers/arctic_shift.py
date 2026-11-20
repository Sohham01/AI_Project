import logging
from typing import Any

from app.schemas.ingestion import CommentData, PostData
from app.services.ingestion.arctic_shift import ArcticShiftClient
from app.services.ingestion.arctic_shift_normalizer import ArcticShiftNormalizer
from app.services.ingestion.providers.base import IngestionProvider, ProviderRequestError


logger = logging.getLogger(__name__)


class ArcticShiftProvider(IngestionProvider):
    """Single-request snapshots; neither operation promises archive completeness."""

    def __init__(self) -> None:
        self.client = ArcticShiftClient()
        self.normalizer = ArcticShiftNormalizer()
        self.last_search_complete = False
        self.last_comments_complete = False
        self.last_comments_truncated = False

    def search_posts(self, **kwargs: Any) -> list[PostData]:
        """Return at most ``limit`` distinct posts from one server-side page.

        A short/empty page is not proof that an archive/time window is complete.
        In particular, no timestamp pagination is attempted around tied dates.
        """
        allowed = {"subreddit", "query", "title", "selftext", "limit", "after", "before", "sort", "timeout"}
        if kwargs.keys() - allowed:
            raise ValueError("unsupported Arctic Shift search parameter")
        subreddit = kwargs.get("subreddit")
        limit = ArcticShiftClient._limit(kwargs.get("limit", 25), ArcticShiftClient.MAX_POSTS)
        search_kwargs: dict[str, Any] = {
            "subreddit": subreddit,
            "query": kwargs.get("query"),
            "limit": limit,
            "after": kwargs.get("after", "30d"),
            "before": kwargs.get("before"),
            "sort": kwargs.get("sort", "desc"),
            "timeout": kwargs.get("timeout"),
        }
        if "title" in kwargs:
            search_kwargs["title"] = kwargs.get("title")
        if "selftext" in kwargs:
            search_kwargs["selftext"] = kwargs.get("selftext")
        raw_response = self.client.search_posts(**search_kwargs)
        raw_posts = ArcticShiftClient.response_data(raw_response)
        posts: dict[str, PostData] = {}
        for raw_post in raw_posts:
            post = self.normalizer.normalize_post(raw_post)
            if subreddit is not None and (
                not isinstance(subreddit, str)
                or post.subreddit.casefold() != subreddit.removeprefix("r/").casefold()
            ):
                raise ProviderRequestError("invalid_payload")
            previous = posts.get(post.reddit_id)
            if previous is not None and previous != post:
                raise ProviderRequestError("invalid_payload")
            posts.setdefault(post.reddit_id, post)
        if len(raw_posts) >= limit:
            logger.warning("Arctic Shift returned a bounded post page; window completeness is unknown")
        return list(posts.values())[:limit]

    def get_post(self, reddit_id: str) -> PostData:
        requested = ArcticShiftClient._id(reddit_id, "t3")
        raw_posts = ArcticShiftClient.response_data(self.client.get_post(reddit_id))
        if not raw_posts:
            raise ValueError("Arctic Shift post not found")
        if len(raw_posts) != 1:
            raise ProviderRequestError("invalid_payload")
        post = self.normalizer.normalize_post(raw_posts[0])
        if post.reddit_id != requested:
            raise ProviderRequestError("invalid_payload")
        return post

    def get_comments(
        self, post_id: str, timeout: float | None = None, limit: int = 100
    ) -> list[CommentData]:
        """Flatten one bounded tree without expanding collapsed ``more`` nodes.

        Validate every returned node, including deleted parents and nodes beyond
        the local output cap. Parent identities are retained even when their
        deleted bodies are omitted. ``last_comments_truncated`` and a warning
        expose observed collapsing/capping; neither operation proves completeness.
        """
        requested = ArcticShiftClient._id(post_id, "t3")
        ArcticShiftClient._limit(limit, ArcticShiftClient.MAX_COMMENTS)
        self.last_comments_truncated = False
        raw_nodes = ArcticShiftClient.response_data(
            self.client.get_comments_tree(post_id, limit=limit, timeout=timeout)
        )
        # Keep raw parent fullnames separately: the canonical model represents a
        # t3 parent as None, so it cannot itself establish request provenance.
        records: dict[str, tuple[CommentData, str, bool]] = {}
        stack = [(node, None, False) for node in reversed(raw_nodes)]
        active: set[int] = set()
        while stack:
            node, expected_parent, exiting = stack.pop()
            if exiting:
                active.remove(id(node))
                continue
            if not isinstance(node, dict) or id(node) in active:
                raise ProviderRequestError("invalid_payload")
            kind, data = node.get("kind"), node.get("data")
            if kind not in ("t1", "more") or not isinstance(data, dict):
                raise ProviderRequestError("invalid_payload")
            has_link = "link_id" in data
            if has_link and self.normalizer._identity(data["link_id"], "t3") != requested:
                raise ProviderRequestError("invalid_payload")
            if kind == "more":
                children = data.get("children")
                if not isinstance(children, list):
                    raise ProviderRequestError("invalid_payload")
                for child in children:
                    self.normalizer._identity(child, "t1")
                parent = data.get("parent_id")
                if parent is not None:
                    if not isinstance(parent, str) or not parent.startswith(("t1_", "t3_")):
                        raise ProviderRequestError("invalid_payload")
                    self.normalizer._identity(parent, parent[:2])
                    if expected_parent is not None and parent != f"t1_{expected_parent}":
                        raise ProviderRequestError("invalid_payload")
                    if parent.startswith("t3_"):
                        if self.normalizer._identity(parent, "t3") != requested:
                            raise ProviderRequestError("invalid_payload")
                self.last_comments_truncated = True
                continue
            comment = self.normalizer.normalize_comment(data)
            parent = data["parent_id"]
            if expected_parent is not None and parent != f"t1_{expected_parent}":
                raise ProviderRequestError("invalid_payload")
            if parent.startswith("t3_") and self.normalizer._identity(parent, "t3") != requested:
                raise ProviderRequestError("invalid_payload")
            previous = records.get(comment.reddit_id)
            if previous is not None:
                if previous[:2] != (comment, parent):
                    raise ProviderRequestError("invalid_payload")
                has_link = has_link or previous[2]
            records[comment.reddit_id] = (comment, parent, has_link)
            replies = data.get("replies")
            if replies is None or replies == "":
                continue
            if (
                not isinstance(replies, dict)
                or replies.get("kind", "Listing") != "Listing"
                or not isinstance(replies.get("data"), dict)
                or not isinstance(replies["data"].get("children"), list)
            ):
                raise ProviderRequestError("invalid_payload")
            active.add(id(node))
            stack.append((node, expected_parent, True))
            stack.extend(
                (child, comment.reddit_id, False)
                for child in reversed(replies["data"]["children"])
            )

        # Resolve flattened t1-parent chains too. An orphan is only safe when
        # its own link_id explicitly binds it to the requested post.
        verified: set[str] = set()
        for identity in records:
            path: set[str] = set()
            current = identity
            while current not in verified:
                if current in path:
                    raise ProviderRequestError("invalid_payload")
                path.add(current)
                _, parent, has_link = records[current]
                if parent.startswith("t3_"):
                    break
                ancestor = self.normalizer._identity(parent, "t1")
                if ancestor not in records:
                    if not has_link:
                        raise ProviderRequestError("invalid_payload")
                    break
                current = ancestor
            verified.update(path)

        comments = [
            comment for comment, _, _ in records.values()
            if comment.body not in {"[deleted]", "[removed]", ""}
        ]
        if len(comments) > limit:
            self.last_comments_truncated = True
        if self.last_comments_truncated:
            logger.warning("Arctic Shift comment snapshot is incomplete (collapsed branches or output cap)")
        return comments[:limit]
