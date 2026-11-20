import calendar
from datetime import datetime, timedelta, timezone
import math
import re
from typing import Any

import requests

from app.services.ingestion.providers.base import ProviderRequestError


class ArcticShiftClient:
    BASE_URL = "https://arctic-shift.photon-reddit.com"
    MAX_POSTS = 100
    MAX_COMMENTS = 25000

    def __init__(self, timeout: int = 60) -> None:
        self.timeout = self._timeout(timeout)
        self.last_ratelimit_reset: float | None = None
        self.last_ratelimit_remaining: int | None = None
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "IIM-Community-Intelligence/1.0",
            }
        )

    @staticmethod
    def _timeout(value: Any) -> float:
        try:
            finite = isinstance(value, (int, float)) and math.isfinite(value)
        except OverflowError:
            finite = False
        if (
            isinstance(value, bool)
            or not finite
            or value <= 0
        ):
            raise ValueError("timeout must be a finite positive number")
        return float(value)

    @staticmethod
    def _limit(value: Any, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"limit must be an integer between 1 and {maximum}")
        return value

    @staticmethod
    def _id(value: Any, kind: str) -> str:
        if not isinstance(value, str):
            raise ValueError("invalid Reddit identity")
        clean = value.removeprefix(f"{kind}_")
        if not re.fullmatch(r"[a-z0-9]+", clean):
            raise ValueError("invalid Reddit identity")
        return clean

    @staticmethod
    def _date(value: Any, now: datetime) -> tuple[str | None, datetime | None]:
        """Validate documented date forms and compare all bounds in UTC.

        Relative month/year offsets use calendar arithmetic. Relative strings
        remain relative on the wire; absolute values are serialized in UTC.
        """
        if value is None:
            return None, None
        if isinstance(value, bool):
            raise ValueError("invalid date bound")
        try:
            if isinstance(value, datetime):
                parsed = value
            elif isinstance(value, (int, float)) or (
                isinstance(value, str)
                and re.fullmatch(r"-?\d+(?:\.\d+)?", value)
                and len(value.lstrip("-")) != 4
            ):
                epoch = float(value)
                if not math.isfinite(epoch):
                    raise ValueError
                if abs(epoch) >= 100_000_000_000:
                    epoch /= 1000
                parsed = datetime.fromtimestamp(epoch, tz=timezone.utc)
            elif isinstance(value, str):
                relative = re.fullmatch(
                    r"(\d+)(year|years|month|months|m|week|weeks|w|day|days|d|hour|hours|h|min|minute|minutes|second|seconds|s)",
                    value,
                )
                if relative:
                    amount, unit = int(relative[1]), relative[2]
                    if unit in {"year", "years", "month", "months", "m"}:
                        months = amount * 12 if unit in {"year", "years"} else amount
                        year, month = divmod(now.year * 12 + now.month - 1 - months, 12)
                        month += 1
                        parsed = now.replace(
                            year=year, month=month,
                            day=min(now.day, calendar.monthrange(year, month)[1]),
                        )
                    else:
                        seconds = {
                            "week": 604800, "weeks": 604800, "w": 604800,
                            "day": 86400, "days": 86400, "d": 86400,
                            "hour": 3600, "hours": 3600, "h": 3600,
                            "min": 60, "minute": 60, "minutes": 60,
                            "second": 1, "seconds": 1, "s": 1,
                        }[unit]
                        parsed = now - timedelta(seconds=amount * seconds)
                    return value, parsed
                if re.fullmatch(r"\d{4}", value):
                    value += "-01-01"
                elif re.fullmatch(r"\d{4}-\d{2}", value):
                    value += "-01"
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            else:
                raise ValueError
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            parsed = parsed.astimezone(timezone.utc)
            return parsed.isoformat().replace("+00:00", "Z"), parsed
        except (ValueError, OverflowError, OSError):
            raise ValueError("invalid date bound") from None

    @staticmethod
    def response_data(payload: Any) -> list[dict[str, Any]]:
        if (
            not isinstance(payload, dict)
            or "data" not in payload
            or not isinstance(payload["data"], list)
            or any(not isinstance(row, dict) for row in payload["data"])
        ):
            raise ProviderRequestError("invalid_payload")
        return payload["data"]

    def _get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        request_timeout = self.timeout if timeout is None else self._timeout(timeout)
        try:
            response = self.session.get(
                f"{self.BASE_URL}{endpoint}", params=params, timeout=request_timeout,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            if isinstance(exc, requests.Timeout):
                category = "timeout"
            elif isinstance(exc, requests.ConnectionError):
                category = "connection"
            else:
                category = "transport"
            raise ProviderRequestError(category) from None
        if isinstance(response.status_code, bool) or not isinstance(response.status_code, int):
            raise ProviderRequestError("invalid_payload")
        self._record_ratelimit_headers(response)
        if not 200 <= response.status_code < 300:
            if response.status_code == 422 and self._is_server_timeout_response(response):
                raise ProviderRequestError("timeout", 422)
            raise ProviderRequestError("http_status", response.status_code)
        try:
            payload = response.json()
        except ValueError:
            raise ProviderRequestError("invalid_json") from None
        self.response_data(payload)
        return payload

    def _record_ratelimit_headers(self, response: Any) -> None:
        """Record Arctic Shift rate-limit headers safely without failing on mocks."""
        headers = getattr(response, "headers", None)
        if not isinstance(headers, dict) and not hasattr(headers, "get"):
            return
        raw_reset = headers.get("x-ratelimit-reset") or headers.get("X-RateLimit-Reset")
        if raw_reset is not None:
            try:
                val = float(raw_reset)
                if math.isfinite(val) and 0.0 <= val <= 300.0:
                    self.last_ratelimit_reset = val
            except (TypeError, ValueError):
                pass
        raw_rem = headers.get("x-ratelimit-remaining") or headers.get("X-RateLimit-Remaining")
        if raw_rem is not None:
            try:
                rem = int(raw_rem)
                if rem >= 0:
                    self.last_ratelimit_remaining = rem
            except (TypeError, ValueError):
                pass

    @staticmethod
    def _is_server_timeout_response(response: Any) -> bool:
        """Detect Arctic Shift's documented HTTP 422 server-side timeout/throttle response.

        Arctic Shift returns ``HTTP 422 {"data": null, "error": "Timeout. Maybe slow down a bit"}``
        when a search query exceeds its backend execution budget or hits sliding-window
        pacing limits. Inspects ``response.text`` without invoking ``response.json()``
        and never echoes any response content into exception messages.
        """
        raw_text = getattr(response, "text", None)
        if not isinstance(raw_text, str):
            return False
        lowered = raw_text.lower()
        return any(tok in lowered for tok in ("timeout", "timed out", "slow down", "rate limit"))

    def search_posts(
        self,
        *,
        subreddit: str | None = None,
        query: str | None = None,
        title: str | None = None,
        selftext: str | None = None,
        limit: int = 25,
        after: str | None = "30d",
        before: str | None = None,
        sort: str = "desc",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Fetch one bounded page, never a complete archive/window crawl.

        The upstream API README documents ``query`` (title/selftext), ``title``,
        and ``selftext`` keyword search, requiring subreddit or author scope.
        This client supports the subreddit scope.
        """
        self._limit(limit, self.MAX_POSTS)
        if sort not in ("asc", "desc"):
            raise ValueError("sort must be asc or desc")
        if subreddit is not None:
            if not isinstance(subreddit, str):
                raise ValueError("invalid subreddit")
            subreddit = subreddit.removeprefix("r/")
            if not subreddit or not re.fullmatch(r"[A-Za-z0-9_.]+", subreddit):
                raise ValueError("invalid subreddit")
        for field_name, field_val in (("query", query), ("title", title), ("selftext", selftext)):
            if field_val is not None and not isinstance(field_val, str):
                raise ValueError(f"{field_name} must be text")
        query = query.strip() if query is not None else None
        title = title.strip() if title is not None else None
        selftext = selftext.strip() if selftext is not None else None
        if (query or title or selftext) and not subreddit:
            raise ValueError("keyword search requires a subreddit")
        now = datetime.now(timezone.utc)
        after_param, after_date = self._date(after, now)
        before_param, before_date = self._date(before, now)
        if after_date is not None and before_date is not None and after_date >= before_date:
            raise ValueError("after must precede before")
        if timeout is not None:
            self._timeout(timeout)
        params: dict[str, Any] = {"limit": limit, "sort": sort}
        if subreddit:
            params["subreddit"] = subreddit
        if query:
            params["query"] = query
        if title:
            params["title"] = title
        if selftext:
            params["selftext"] = selftext
        if after_param is not None:
            params["after"] = after_param
        if before_param is not None:
            params["before"] = before_param
        # Every field is listed in the upstream selectable-fields table.
        params["fields"] = "id,title,selftext,subreddit,created_utc,score,num_comments,url"
        return self._get("/api/posts/search", params=params, timeout=timeout)

    def get_post(self, reddit_id: str) -> dict[str, Any]:
        return self._get("/api/posts/ids", params={"ids": self._id(reddit_id, "t3")})

    def get_comments_tree(
        self,
        post_id: str,
        limit: int = 100,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        clean_post_id = self._id(post_id, "t3")
        self._limit(limit, self.MAX_COMMENTS)
        if timeout is not None:
            self._timeout(timeout)
        return self._get(
            "/api/comments/tree",
            params={"link_id": f"t3_{clean_post_id}", "limit": limit},
            timeout=timeout,
        )

    def search_comments(
        self,
        *,
        subreddit: str | None = None,
        query: str | None = None,
        link_id: str | None = None,
        limit: int = 25,
        after: str | None = "30d",
        before: str | None = None,
        sort: str = "desc",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Fetch one bounded page of comments from ``/api/comments/search``."""
        self._limit(limit, self.MAX_POSTS)
        if sort not in ("asc", "desc"):
            raise ValueError("sort must be asc or desc")
        if subreddit is not None:
            if not isinstance(subreddit, str):
                raise ValueError("invalid subreddit")
            subreddit = subreddit.removeprefix("r/")
            if not subreddit or not re.fullmatch(r"[A-Za-z0-9_.]+", subreddit):
                raise ValueError("invalid subreddit")
        if query is not None and not isinstance(query, str):
            raise ValueError("query must be text")
        query = query.strip() if query is not None else None
        if query and not (subreddit or link_id):
            raise ValueError("keyword search requires a subreddit or link_id")
        clean_link = self._id(link_id, "t3") if link_id is not None else None
        now = datetime.now(timezone.utc)
        after_param, after_date = self._date(after, now)
        before_param, before_date = self._date(before, now)
        if after_date is not None and before_date is not None and after_date >= before_date:
            raise ValueError("after must precede before")
        if timeout is not None:
            self._timeout(timeout)
        params: dict[str, Any] = {"limit": limit, "sort": sort}
        if subreddit:
            params["subreddit"] = subreddit
        if query:
            params["body"] = query
        if clean_link:
            params["link_id"] = f"t3_{clean_link}"
        if after_param is not None:
            params["after"] = after_param
        if before_param is not None:
            params["before"] = before_param
        return self._get("/api/comments/search", params=params, timeout=timeout)
