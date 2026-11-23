"""HTTP security header middleware shared by Docker and Vercel FastAPI apps."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

STUDIO_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'; "
    "form-action 'self'"
)

DOCS_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https://fastapi.tiangolo.com; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'"
)

BASE_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach defensive browser security headers to every HTTP response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        for header, value in BASE_SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)

        path = request.url.path
        csp = DOCS_CSP if path.startswith(("/docs", "/redoc")) else STUDIO_CSP
        response.headers.setdefault("Content-Security-Policy", csp)
        if path == "/app" or path.startswith(("/app/", "/assets/")):
            response.headers.setdefault("Cache-Control", "no-cache, must-revalidate")
        elif request.method == "GET" and path.startswith("/api/demo/") and path != "/api/demo/health":
            response.headers.setdefault(
                "Cache-Control",
                "public, max-age=300, stale-while-revalidate=3600",
            )
        return response
