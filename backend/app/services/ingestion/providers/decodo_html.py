"""Offline parser for the HTML returned by Decodo's Universal scraper.

The Universal scraper is intentionally treated as an HTML boundary.  This
module does not issue requests and does not depend on a particular browser
runtime.  It understands the stable ``shreddit-post``/``shreddit-comment``
attributes used by the synthetic fixtures and a small set of explicit
``data-*`` aliases for normalized HTML snapshots.

Live HTML is not assumed to be complete: omitted parent nodes, load-more
markers, missing optional metadata, and output caps are surfaced as static
diagnostics rather than repaired or presented as a complete thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
import re
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

from app.schemas.ingestion import CommentData, PostData


_ID = re.compile(r"[A-Za-z0-9]+")
_INT = re.compile(r"-?\d+")
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
)


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str]
    children: list["_Node"] = field(default_factory=list)
    text: list[str] = field(default_factory=list)

    def descendants(self) -> Iterable["_Node"]:
        for child in self.children:
            yield child
            yield from child.descendants()

    def text_content(self) -> str:
        parts = list(self.text)
        for child in self.children:
            parts.append(child.text_content())
        return "".join(parts)


class _TreeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("__root__", {})
        self._stack = [self.root]
        self.malformed = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        node = _Node(tag.lower(), {name.lower(): value or "" for name, value in attrs})
        self._stack[-1].children.append(node)
        if node.tag not in _VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        self._stack[-1].children.append(
            _Node(tag.lower(), {name.lower(): value or "" for name, value in attrs})
        )

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if not any(node.tag == tag for node in self._stack[1:]):
            self.malformed = True
            return
        while self._stack[-1].tag != tag:
            self._stack.pop()
            self.malformed = True
        self._stack.pop()

    def handle_data(self, data: str) -> None:
        self._stack[-1].text.append(data)

    def close(self) -> None:
        super().close()
        if len(self._stack) != 1:
            self.malformed = True
            self._stack = [self.root]


class ParsedRedditHtml(list[PostData]):
    """List-compatible parsed posts carrying safe completeness diagnostics."""

    def __init__(self, posts: Iterable[PostData] = (), *, diagnostics: Iterable[str] = ()) -> None:
        super().__init__(posts)
        self.diagnostics = tuple(sorted(set(diagnostics)))


def _first_attr(node: _Node, names: Iterable[str]) -> Optional[str]:
    for name in names:
        value = node.attrs.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _is_nonempty(value: Optional[str]) -> bool:
    return value is not None and bool(value.strip())


def _bare_id(value: str, kind: str) -> str:
    clean = value.strip()
    prefix = f"{kind}_"
    if clean.startswith(prefix):
        clean = clean[len(prefix):]
    if not _ID.fullmatch(clean):
        raise ValueError("invalid Reddit identifier")
    return clean.lower()


def _timestamp(value: Optional[str]) -> datetime:
    if value is None or not value.strip():
        raise ValueError("missing timestamp")
    raw = value.strip()
    if _INT.fullmatch(raw):
        epoch = int(raw)
        return datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _integer(value: Optional[str], *, default: int, diagnostics: set[str], missing_code: str) -> int:
    if value is None or not value.strip():
        diagnostics.add(missing_code)
        return default
    if not _INT.fullmatch(value.strip()):
        raise ValueError("invalid integer")
    parsed = int(value.strip())
    if parsed < 0 and default >= 0:
        raise ValueError("invalid nonnegative integer")
    return parsed


def _url(value: Optional[str]) -> Optional[str]:
    if value is None or not value.strip():
        return None
    value = value.strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or any(char.isspace() for char in value):
        raise ValueError("invalid source URL")
    return value


def _field_text(node: _Node, *, attrs: Iterable[str], selectors: Iterable[tuple[str, str]], tags: Iterable[str] = ()) -> Optional[str]:
    value = _first_attr(node, attrs)
    if value is not None:
        return value
    wanted = set(tags)
    for child in node.descendants():
        if any(child.attrs.get(name) == expected for name, expected in selectors) or child.tag in wanted:
            text = child.text_content().strip()
            if text:
                return text
    return None


def _post_id(node: _Node) -> Optional[str]:
    value = _first_attr(
        node,
        ("data-reddit-post-id", "post-id", "data-post-id", "thing-id", "thingid", "data-fullname", "id"),
    )
    if value is None:
        return None
    try:
        return _bare_id(value, "t3")
    except ValueError:
        return None


def _comment_id(node: _Node) -> Optional[str]:
    value = _first_attr(
        node,
        ("data-reddit-comment-id", "comment-id", "data-comment-id", "thing-id", "thingid", "data-fullname", "id"),
    )
    if value is None:
        return None
    try:
        return _bare_id(value, "t1")
    except ValueError:
        return None


def _subreddit(node: _Node) -> Optional[str]:
    value = _first_attr(node, ("subreddit-prefixed-name", "subreddit", "data-subreddit"))
    if value is None:
        return None
    value = value.strip().removeprefix("r/").removeprefix("/r/")
    if not value or any(char.isspace() for char in value) or "/" in value:
        raise ValueError("invalid subreddit")
    return value


def _parent(value: Optional[str], post_id: str, diagnostics: set[str]) -> Optional[str]:
    if value is None or not value.strip():
        diagnostics.add("comment_parent_missing")
        return None
    raw = value.strip()
    if raw.startswith("t3_"):
        if _bare_id(raw, "t3") != post_id:
            raise ValueError("comment belongs to another post")
        return None
    if raw.startswith("t1_"):
        return _bare_id(raw, "t1")
    raise ValueError("invalid comment parent")


def _is_post_node(node: _Node) -> bool:
    return node.tag == "shreddit-post" or _post_id(node) is not None and (
        "data-reddit-post-id" in node.attrs or node.attrs.get("id", "").startswith("t3_")
    )


def _is_comment_node(node: _Node) -> bool:
    return node.tag == "shreddit-comment" or _comment_id(node) is not None and (
        "data-reddit-comment-id" in node.attrs or node.attrs.get("id", "").startswith("t1_")
    )


def _has_partial_marker(node: _Node) -> bool:
    for current in (node, *node.descendants()):
        for name in ("data-more-comments", "data-comments-more", "data-partial-thread", "data-after", "after"):
            value = current.attrs.get(name)
            if value is not None and value.strip().lower() not in {"", "0", "false", "none", "null"}:
                return True
        if current.attrs.get("data-testid", "").lower() in {"load-more-comments", "comments-more"}:
            return True
    return False


def _extract_comment(node: _Node, post_id: str, diagnostics: set[str]) -> CommentData:
    comment_id = _comment_id(node)
    if comment_id is None:
        raise ValueError("missing comment identifier")
    parent = _parent(
        _first_attr(node, ("parent-id", "parent_id", "parentid", "data-parent-id")),
        post_id,
        diagnostics,
    )
    body = _field_text(
        node,
        attrs=("comment-body", "body", "data-comment-body"),
        selectors=(("slot", "comment"), ("data-field", "comment"), ("data-field", "body")),
    )
    if not _is_nonempty(body):
        raise ValueError("missing comment body")
    assert body is not None
    created_at = _timestamp(_first_attr(node, ("created-timestamp", "created-at", "created_utc", "data-created-at")))
    score = _integer(
        _first_attr(node, ("score", "score-count", "data-score")),
        default=0,
        diagnostics=diagnostics,
        missing_code="comment_score_missing",
    )
    return CommentData(
        reddit_id=comment_id,
        body=body,
        created_at=created_at,
        score=score,
        parent_reddit_id=parent,
    )


def _extract_post(node: _Node, *, comment_limit: int, diagnostics: set[str]) -> PostData:
    post_id = _post_id(node)
    if post_id is None:
        raise ValueError("missing post identifier")
    subreddit = _subreddit(node)
    if subreddit is None:
        raise ValueError("missing subreddit")
    title = _field_text(
        node,
        attrs=("post-title", "title", "data-title"),
        selectors=(("slot", "title"), ("data-field", "title")),
        tags=("h1",),
    )
    if not _is_nonempty(title):
        raise ValueError("missing post title")
    assert title is not None
    body = _field_text(
        node,
        attrs=("post-body", "selftext", "data-body"),
        selectors=(("slot", "text-body"), ("data-field", "body"), ("data-field", "text-body")),
    )
    if body == "":
        body = None
    created_at = _timestamp(_first_attr(node, ("post-created", "created-timestamp", "created-at", "created_utc", "data-created-at")))
    score = _integer(
        _first_attr(node, ("score", "score-count", "data-score")),
        default=0,
        diagnostics=diagnostics,
        missing_code="post_score_missing",
    )
    num_comments = _integer(
        _first_attr(node, ("comment-count", "num-comments", "data-comment-count")),
        default=0,
        diagnostics=diagnostics,
        missing_code="post_comment_count_missing",
    )
    url = _url(_first_attr(node, ("permalink", "url", "content-href", "data-url")))

    comments: list[CommentData] = []
    seen: dict[str, CommentData] = {}
    for comment_node in (child for child in node.descendants() if _is_comment_node(child)):
        try:
            comment = _extract_comment(comment_node, post_id, diagnostics)
        except (TypeError, ValueError):
            diagnostics.add("comment_record_rejected")
            continue
        previous = seen.get(comment.reddit_id)
        if previous is not None:
            if previous != comment:
                raise ValueError("conflicting duplicate comment")
            diagnostics.add("duplicate_comments_removed")
            continue
        seen[comment.reddit_id] = comment
        comments.append(comment)

    if _has_partial_marker(node):
        diagnostics.add("comments_more_available")
    if len(comments) > comment_limit:
        comments = comments[:comment_limit]
        diagnostics.add("comments_truncated_by_limit")
    returned_ids = {comment.reddit_id for comment in comments}
    if any(comment.parent_reddit_id and comment.parent_reddit_id not in returned_ids for comment in comments):
        diagnostics.add("comment_parent_not_returned")
    if num_comments > len(comments):
        diagnostics.add("fewer_comments_than_reported")
    return PostData(
        reddit_id=post_id,
        title=title,
        body=body,
        subreddit=subreddit,
        created_at=created_at,
        score=score,
        num_comments=num_comments,
        url=url,
        comments=comments,
    )


def parse_reddit_html(
    html: str,
    *,
    max_posts: int = 20,
    comment_limit: int = 25,
) -> ParsedRedditHtml:
    """Parse a bounded HTML snapshot into normalized ``PostData`` records."""

    if not isinstance(html, str) or not html.strip():
        raise ValueError("empty HTML response")
    if isinstance(max_posts, bool) or not isinstance(max_posts, int) or not 1 <= max_posts <= 200:
        raise ValueError("max_posts must be between 1 and 200")
    if isinstance(comment_limit, bool) or not isinstance(comment_limit, int) or not 0 <= comment_limit <= 2000:
        raise ValueError("comment_limit must be between 0 and 2000")

    parser = _TreeParser()
    try:
        parser.feed(html)
        parser.close()
    except (TypeError, ValueError, RecursionError):
        raise ValueError("malformed HTML response") from None

    diagnostics: set[str] = set()
    if parser.malformed:
        diagnostics.add("html_structure_incomplete")
    post_nodes = [node for node in parser.root.descendants() if _is_post_node(node)]
    if not post_nodes:
        raise ValueError("no Reddit post records found")

    posts: list[PostData] = []
    seen: dict[str, PostData] = {}
    for node in post_nodes:
        try:
            post = _extract_post(node, comment_limit=comment_limit, diagnostics=diagnostics)
        except (TypeError, ValueError):
            diagnostics.add("post_record_rejected")
            continue
        previous = seen.get(post.reddit_id)
        if previous is not None:
            if previous != post:
                raise ValueError("conflicting duplicate post")
            diagnostics.add("duplicate_posts_removed")
            continue
        seen[post.reddit_id] = post
        posts.append(post)
        if len(posts) >= max_posts:
            if len(post_nodes) > max_posts:
                diagnostics.add("posts_truncated_by_limit")
            break
    if not posts:
        raise ValueError("no valid Reddit post records found")
    return ParsedRedditHtml(posts, diagnostics=diagnostics)


__all__ = ["ParsedRedditHtml", "parse_reddit_html"]
