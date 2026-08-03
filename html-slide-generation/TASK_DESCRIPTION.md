# Task: explain-databricks

Create an introductory slide deck that explains **what Databricks is** to a newcomer.
This file is the CANONICAL task description — it is copied byte-for-byte into every
candidate's working directory as `instructions.txt`, so every candidate reads the exact
same brief (the fairness rule of this benchmark).

## Audience
A technical decision-maker who is new to Databricks — a data engineer, analytics lead, or
engineering manager who knows general data/cloud concepts but has not used the Databricks
platform. The deck should be informative and credible, not a sales pitch: explain the
*what* and *why*, with enough concrete detail to be useful.

## Goal
By the end of the deck, the reader should understand:
- What problem the Databricks Lakehouse Platform solves (unifying data engineering,
  analytics, and AI on one platform).
- The core building blocks and how they fit together.
- Where AI / ML — including the Foundation Model API — fits in.
- That it runs across the major clouds.

## Style
- Clear titles, short bullet points, readable typography. One idea per slide.
- A title slide and a short closing/summary slide are encouraged (they count toward the
  slide total).
- Prefer a clean, professional look. Inline CSS only (see the format contract below).
- No speaker notes required; the slides themselves should stand alone.

## Hard format contract (the grader checks these mechanically)
- Write EXACTLY ONE file: `./slides.html` (in this working directory).
- It MUST be a single self-contained HTML document:
  - Starts with `<!DOCTYPE html>` and has one `<html>`, `<head>`, and `<body>`.
  - All CSS is INLINE in a `<style>` tag. Do NOT link external stylesheets over http(s).
    (reveal.js from a CDN is tolerated but discouraged — prefer a fully offline file.
    External refs are flagged as a warning by the grader.)
  - Renders correctly when opened via `file://` with no network for core content.
- Slide structure: use EITHER
  - vanilla slides: each slide is an element with `class="slide"`, OR
  - reveal.js: each slide is a `<section>` element.
  The grader counts whichever it finds.
- **Slide count: between 8 and 10 slides (inclusive).**
- Required topics — every deck MUST cover each of these (the grader does a keyword-coverage
  check on the rendered text):
  - Lakehouse architecture (unifying data lake + data warehouse)
  - Unity Catalog (governance, lineage, access control)
  - Delta Lake (ACID transactions on the lake)
  - Apache Spark / distributed compute
  - Databricks SQL / data warehousing
  - Machine learning, Mosaic AI, and the Foundation Model API (FMAPI)
  - Workflows / job orchestration
  - Multi-cloud deployment (AWS, Azure, GCP)

## Reminders
- Work offline for the core content — do not depend on fetching data at render time.
- Keep iterating until `./slides.html` exists and satisfies the contract above.
