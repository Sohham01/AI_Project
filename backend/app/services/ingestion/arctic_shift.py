from typing import Any

import requests


class ArcticShiftClient:
    BASE_URL = "https://arctic-shift.photon-reddit.com"

    def __init__(self, timeout: int = 60) -> None:
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "IIM-Community-Intelligence/1.0",
            }
        )

    def _get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        url = f"{self.BASE_URL}{endpoint}"

        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout,
        )

        if not response.ok:
            raise RuntimeError(
                f"Arctic Shift request failed\n"
                f"Status: {response.status_code}\n"
                f"URL: {response.url}\n"
                f"Response: {response.text}"
            )

        return response.json()

    def search_posts(
        self,
        *,
        subreddit: str,
        limit: int = 5,
        after: str | None = "30d",
        before: str | None = None,
        sort: str = "desc",
    ) -> dict[str, Any]:

        params: dict[str, Any] = {
            "subreddit": subreddit,
            "limit": limit,
            "sort": sort,
        }

        if after:
            params["after"] = after

        if before:
            params["before"] = before

        # Arctic Shift API may support a fields parameter.
        # If not, it will just be ignored.
        params["fields"] = "id,title,selftext,subreddit,created_utc,score,num_comments,url"

        return self._get(
            "/api/posts/search",
            params=params,
        )

    def get_post(self, reddit_id: str) -> dict[str, Any]:
        return self._get(
            "/api/posts/ids",
            params={"ids": reddit_id},
        )

    def get_comments_tree(
        self,
        post_id: str,
        limit: int = 9999,
    ) -> dict[str, Any]:

        clean_post_id = post_id.removeprefix("t3_")

        return self._get(
            "/api/comments/tree",
            params={
                "link_id": f"t3_{clean_post_id}",
                "limit": limit,
            },
        )