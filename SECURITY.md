# Security and privacy

Do not commit datasets, raw captures, model checkpoints, prediction exports, experiment logs, manuscripts, credentials, or machine-specific absolute paths. The repository ignore rules cover the expected local locations and common model formats, but contributors must still inspect staged changes before every push.

Dataset CSV paths are treated as untrusted relative paths and must remain inside their configured image root. Generated prediction tables use dataset-relative identifiers instead of workstation paths.

Please report a suspected security or privacy issue privately to the repository owner rather than opening a public issue containing sensitive material.
