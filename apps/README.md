# apps/ — Phase 3

The web application (FastAPI backend + React/MapLibre frontend) is built in **Phase 3**, only
after the Phase 1 vertical slice runs end to end. It is a second front-end onto `sonaris run`
and never reimplements pipeline logic. See `plan.md` §8 and `SONARIS_Build_Layout.md` §11.

Planned:

```
apps/api/    FastAPI · SQLite · worker (subprocess) · SSE progress · render
apps/web/    Vite · React · MapLibre · linked selection · evidence panel · operator review
```
