from app.services.ingestion.reddit import RedditIngestion


def main() -> None:

    ingestion = RedditIngestion()

    posts = ingestion.fetch_posts(
        query="IIM",
        subreddit="CATpreparation",
        limit=10,
    )

    print(f"Posts found: {len(posts)}")

    for post in posts:

        comments = ingestion.fetch_comments(
            post.reddit_id
        )

        print("\nPOST")
        print("Reddit ID:", post.reddit_id)
        print("Title:", post.title)
        print("\nCOMMENTS:", len(comments))
        print()

        for comment in comments[:10]:
            print("COMMENT")
            print("Reddit ID:", comment.reddit_id)
            print("Parent:", comment.parent_reddit_id)
            print("Score:", comment.score)
            print("Body:", comment.body)
            print()


if __name__ == "__main__":
    main()