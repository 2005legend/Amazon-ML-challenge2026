"""Macro F-beta exactly as defined by the challenge, plus diagnostics."""
import polars as pl


def _per_s1(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> pl.DataFrame:
    pred = pred.select("s1_id", "tgt_id").unique()
    truth = truth.select("s1_id", "tgt_id").unique()
    n_pred = pred.group_by("s1_id").len(name="n_pred")
    n_true = truth.group_by("s1_id").len(name="n_true")
    tp = pred.join(truth, on=["s1_id", "tgt_id"], how="inner").group_by("s1_id").len(name="tp")
    return (pl.DataFrame({"s1_id": s1_ids}).unique()
              .join(n_pred, on="s1_id", how="left")
              .join(n_true, on="s1_id", how="left")
              .join(tp, on="s1_id", how="left")
              .with_columns(pl.col("n_pred", "n_true", "tp").fill_null(0)))


def fbeta_expr(beta: float) -> pl.Expr:
    b2 = beta * beta
    return (pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
              .when(pl.col("tp") == 0).then(0.0)
              .otherwise((1 + b2) * pl.col("tp") / (b2 * pl.col("n_true") + pl.col("n_pred"))))


def macro_fbeta(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series, beta: float = 0.5) -> float:
    return float(_per_s1(pred, truth, s1_ids).select(fbeta_expr(beta).mean()).item())


def per_s1_scores(pred, truth, s1_ids, beta: float = 0.5) -> pl.DataFrame:
    return _per_s1(pred, truth, s1_ids).with_columns(fbeta_expr(beta).alias("f"))


def blocking_ceiling(cands: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series, beta: float = 0.5) -> float:
    """Macro F-beta of a perfect matcher restricted to the candidate set."""
    found = cands.select("s1_id", "tgt_id").unique().join(truth, on=["s1_id", "tgt_id"], how="inner")
    return macro_fbeta(found, truth, s1_ids, beta)
