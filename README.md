# Standalone grading app

Copy this entire folder to another location or machine. It contains the app,
browser assets, trial artifacts, requirements, benchmark scores, and grades.
No other repository folders, network downloads, or pip packages are needed.
A Python 3.9+ interpreter and a web browser are required on the destination.
The trial scripts are review artifacts; the grading app does not execute them.

Run from this folder:

```sh
python3 grading_app.py
```

Open http://127.0.0.1:8080. All grading writes stay inside this folder.

Nathan's login credentials have been removed. Existing grades and their Nathan
attribution are preserved. As Nathan was the only account, this copy initially
has no active logins. Spectator browsing remains available; the existing app
anonymizes grader names for spectators and displays names to logged-in graders.

To enable grading, add a new account to `grader_accounts.csv`, retaining its
`username,password` header. Use a new username and a password you choose.
The app uses plaintext local authentication: run it only on a trusted machine.

`results/` contains independent copies of all original trial artifacts, and
`prompts/trials.json` contains their grading requirements. The original app
and original data have not been changed.
