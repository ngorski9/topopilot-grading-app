# TopoPilot grading app

Grading app from the TopoPilot submission. This app can be used to insepct the grading results from using Claude Code and Codex, including the manual grading and the evaluation using the SciVisAgentBench scores. By logging in as a grader, you also can see what the grading interface is like.

## Prerequisites

No libraries are required. Requires Python 3.9+

## Usage

```
python3 grading_app.py
```

Open http://127.0.0.1:8080. All grading writes stay inside this folder.

### Logging in as a grader

There are currently no active grading credentials, although the previous grading results are present. To enable grading, add a new account to `grader_accounts.csv`, retaining its `username,password` header. The login uses unencrypted login information, so only use it on a trusted machine.

### Artifacts and grading requirements

All independent copies of the trial argifacts are found in `results/`. The prompts along with their requirements can be found in `prompts/trials.json`.
