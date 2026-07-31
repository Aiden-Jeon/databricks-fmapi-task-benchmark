# explain-databricks

Benchmark task: **"generate an HTML slide deck that explains Databricks"**, run across
several candidate models and graded identically.

## Layout
```
explain-databricks/
├── README.md            # this file
├── TASK_DESCRIPTION.md  # canonical shared brief + hard format contract (copied to each candidate)
├── keywords.json        # machine grading config (slide bounds + required topics)
├── opus/                # candidate output dir  (slides.html, instructions.txt, run_meta.json, screenshots/)
├── sol/                 # candidate output dir
└── glm/                 # candidate output dir
```
Each candidate directory is one model/candidate being compared. The candidate name
(`opus`, `sol`, `glm`, …) is arbitrary — add more by running the runner with a new
`--candidate` name. The runner writes a byte-identical `instructions.txt` (copied from
`TASK_DESCRIPTION.md`) and `prompt.txt` into each candidate dir before producing
`slides.html`, so every candidate reads the exact same brief.

## Run one candidate
```bash
# from the repo root
uv run run-task --task explain-databricks --candidate opus \
    --harness direct-fmapi --model databricks-claude-opus-4-1

uv run run-task --task explain-databricks --candidate glm \
    --harness direct-fmapi --model databricks-glm-...

# agent harnesses (candidate name is just a label for the output dir)
uv run run-task --task explain-databricks --candidate opus --harness claude-code
```

## Grade + compare
```bash
uv run grade-task --task explain-databricks
# → comparison table across opus/sol/glm + grade_results.json + gallery/index.html
```
See the repo-root `README.md` for the full playbook and fairness checklist.
