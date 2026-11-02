# Credential handling audit and manual key-revocation checklist

Status of this document: the repository-side audit is **VERIFIED**; the actual
key revocation is **AWAITING USER ACTION**. Nothing in this file should be read as a claim
that any key has been revoked — revocation happens only inside the Google AI Studio /
Google Cloud project that owns the key, and no authorized account integration is available
from this workspace.

No key value, token, or `.env` contents appear in this document, in the commands below, or
anywhere in the repository.

---

## 1. What was audited, and how

Method: pattern search for credential-shaped strings, with results reduced to *metadata*
(file name, line number, token length, character-class shape) rather than values. A token is
never printed, echoed, or copied.

```bash
# from the repository root — -l prints file names only, never the matched text
git grep -l -I -E "AIza[0-9A-Za-z_-]{20,}" HEAD -- .                       # tracked at HEAD
git rev-list --all | while read c; do git grep -l -I -E "AIza[0-9A-Za-z_-]{20,}" $c -- .; done \
  | sed 's/^[0-9a-f]*://' | sort -u                                        # entire history
git grep -l -I -E "AIza[0-9A-Za-z_-]{20,}" -- --cached                     # staged index
git check-ignore -v .env                                                   # ignore rule provenance
git ls-files | grep -E "(^|/)\.env"                                        # tracked env-ish files
```

A second pass (`scratch/check_test_tokens.py`, gitignored) confirmed the shape of each
in-repo match — length, distinct-character count, and whether the surrounding line carries a
placeholder cue — without printing the token itself.

Audited surfaces: tracked files at `HEAD`, every commit reachable on the branch, the staging
index, working-tree untracked files, `.gitignore` rules, `data/` and `eval/runs/` artifacts,
`scratch/`, and the frontend build output.

## 2. Findings

| Check | Result |
| --- | --- |
| `.env` ignored | **Yes.** `.gitignore:32` is `.env`, `:33` is `.env.*`, `:34` re-allows only `.env.example`. `git check-ignore -v .env` confirms the rule is what keeps it out. |
| Tracked env files | `.env.example` and `frontend/.env.example` only — variable *names* and empty/default values, no credentials. |
| Credentials in tracked files at `HEAD` | **None.** The only key-shaped strings are three synthetic test fixtures. |
| Credentials in Git history | **None.** The history-wide scan returns the same three test files, nothing else. |
| Credentials staged for commit | **None.** |
| Corpus / run artifacts leaking secrets | `data/` is gitignored (PDFs and chunk JSON never committed); `eval/runs/*.json` contain questions, chunk ids, page numbers and scores only. `scratch/` is gitignored (`/scratch/` at `.gitignore:57`). |
| Secret values in logs or reports | None found in the docs, the validation report, or the benchmark artifacts. |

The three key-shaped strings in tracked files are **test placeholders, not credentials** —
they exist so the suite can assert that a provider key never reaches a log line or an error
response. They are assigned to variables named `sec_gkey` and `secret_key` in
`backend/tests/test_api_answer.py` (two occurrences) and `backend/tests/test_llm_gemini.py`
(one). Each is a literal the test injects directly into a provider constructor — no test reads
one from the environment, and other cases in the same files delete the environment variable and
assert the "GEMINI_API_KEY is not set" failure path — so the suite passes on a machine with no
real key present. Their placeholder wording is visibly synthetic (`Fake`, `SuperSecret…NotForLogs`),
their lengths are 29–44 characters against the 39-character real-key shape, and the assertion
that follows each injection is that the value never appears in captured logs. They are named
here by variable, never by value.

## 3. What this audit cannot establish

- Whether the key that was pasted into a local terminal during earlier development is still
  active. That state lives only in the Google account that issued it.
- Whether that key was ever used by a third party. Usage logs are in the same account.
- Whether the key appears in shell history, editor backups, clipboard managers, or terminal
  scrollback on this machine. Those are outside the repository and were not read.

The old key is therefore treated as **compromised-by-exposure until the user says otherwise**,
which is the only safe assumption, and rotation is the remedy.

## 4. Manual revocation checklist — `AWAITING USER ACTION`

Perform these in a browser, signed in to the Google account that owns the key. Do not paste
the key into a terminal, an editor, or a chat message while doing it.

1. **Open the key list.** <https://aistudio.google.com/apikey> — AI Studio → *API keys*.
2. **Identify the exposed key without displaying it.** Use the *last four characters* shown in
   the console UI to tell the old key from the new one. Do not copy either value anywhere.
3. **Delete the old key.** Its row menu → *Delete key* → confirm. Deletion is immediate and
   irreversible; if the old key is still wired into anything you want to keep running, create
   its replacement first.
4. **Check the same key in Cloud Console.** <https://console.cloud.google.com/apis/credentials>
   → *Credentials* → *APIs & services*. A key created through Cloud appears there rather than
   (or in addition to) AI Studio; delete or rotate it there too.
5. **Restrict the surviving key.** In Cloud Console, open the key → *Application restrictions*
   (IP address / referrer) and *API restrictions* (only the Gemini Generative Language API).
   Restriction is the durable fix; rotation alone leaves the same exposure class open again.
6. **Look for abuse.** <https://console.cloud.google.com/apis/credentials> → key → *Manage*,
   and the *Metrics* page for the project: request volume and unexpected 403/429 spikes on the
   day the key was exposed are the signal. If anything is unexplained, rotate again.
7. **Confirm billing is capped or the budget alert is set,** so a leaked key cannot generate an
   open-ended bill.
8. **Verify the local application still authenticates with the surviving key only.** The
   surviving key must exist in `.env` (untracked) and nowhere else. Then, with your approval,
   one live `POST /answer` request confirms the path end to end; until then the app's behaviour
   with the new key is verified only as far as *configuration and authentication are loaded
   correctly*, which is a different claim from *a live answer was generated*.
9. **Clean the local trail (optional but recommended).** Clear the terminal scrollback that
   held the paste, and check `~/.bash_history` / PowerShell `ConsoleHost_history.txt` for the
   line that contains it; delete that line if present. This document does not read those files.

After step 3 (or 4) is done, this section can be marked `COMPLETED`. Until then it stays
`AWAITING USER ACTION`, and no PaperTrail documentation may state that the old key is revoked.

## 5. Repository policy this audit supports

README's rule — "No key, token, or credential value belongs in this repository. If a live key
is ever pasted into a terminal or log, treat it as exposed and rotate it" — is the operative
policy. The negative tests in `backend/tests/test_llm_gemini.py` and
`backend/tests/test_api_answer.py` enforce the code half of it: a provider key must not appear
in application logs or in an error response body.

Related: [architecture.md](architecture.md) (where the key is read),
[limitations.md](limitations.md) (provider quota behaviour when a request is refused).
