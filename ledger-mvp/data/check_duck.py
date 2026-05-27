import duckdb
con = duckdb.connect("data/ledger_mvp.duckdb")


df = con.execute("""
    SELECT decision, COUNT(*) AS n
    FROM decision_log
    GROUP BY decision
""").fetchdf()

print(df)

print("=" * 60)


con = duckdb.connect("data/ledger_mvp.duckdb")
df = con.execute("""
    SELECT
        MIN(shadow_ml_score) AS min_score,
        MAX(shadow_ml_score) AS max_score,
        AVG(shadow_ml_score) AS avg_score
    FROM decision_log
""").fetchdf()

print(df)
