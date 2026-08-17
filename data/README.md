# Data

The full ESTA dataset and generated Parquet tables are intentionally not stored in Git.

Download ESTA from [pnxenopoulos/esta](https://github.com/pnxenopoulos/esta) and place
the LAN and Online subsets outside the repository, or pass their location directly to
`src.csdemo.esta_to_tables`.

The local development path used for this project is:

```text
C:\project1\data\esta
```

Only the small synthetic CSV files under `data/sample/` are committed. They are used by
unit tests and do not contain the full ESTA dataset.

ESTA is distributed under its own CC-BY-SA-4.0 license. Review and follow the upstream
license when redistributing derived data.
