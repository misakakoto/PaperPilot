# Data and execution boundary

- Supply only papers and datasets you are entitled to use. The project does not ship a corpus.
- Remote model requests can contain relevant source excerpts. Review the provider,
  endpoint and data scope before enabling API use with private materials.
- Keep keys in process environment variables, an ignored `.env`, or an external
  credential file. Never paste credentials into issues, examples or test fixtures.
- `.paperpilot/`, `outputs/`, papers, databases, weights and local logs are excluded
  from version control. Ignoring a path does not remove previously committed content.
- Inspect generated code and approve each frozen candidate before execution. The
  container boundary reduces host exposure but is not a malicious-code certification.
- Keep the workbench on localhost. Its developer configuration is not a public
  multi-tenant service deployment guide.
- Budget estimates are conservative upper bounds. Model responses and local tool
  execution are separate states; UNKNOWN does not authorize a blind retry.

Run `python scripts/check_release_contents.py` before staging or uploading changes.
It checks the current Git candidate set for selected dangerous files and secret
patterns. It does not audit historical blobs, establish ownership of copied text,
or prove that every possible credential format has been detected.
