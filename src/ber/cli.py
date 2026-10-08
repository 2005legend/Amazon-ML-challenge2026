"""Command line entry point: python -m ber.cli <command> [options]."""
import argparse

from . import config


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ber")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("prepare", help="normalize a split into work/<split>/norm_s*.parquet")
    sp.add_argument("--split", choices=["train", "test"], required=True)
    sb = sub.add_parser("block", help="build raw candidates for a split")
    sb.add_argument("--split", choices=["train", "test"], required=True)
    sub.add_parser("block-report", help="blocking recall on train")
    sub.add_parser("prerank", help="train the pre-ranker on train fold A and prune train+test candidates")
    sf = sub.add_parser("features", help="pairwise features for pruned candidates")
    sf.add_argument("--split", choices=["train", "test"], required=True)
    st = sub.add_parser("tokens", help="add token-change features to existing feature tables")
    st.add_argument("--split", choices=["train", "test"], required=True)
    sub.add_parser("stage1", help="train stage-1 fold models and score train (out-of-fold) and test")
    sub.add_parser("stage2", help="context + sibling features, train stage-2 models, score train and test")
    sub.add_parser("stage3", help="stage-2 model on folds A+B and stage-3 model; score holdout and test (p23)")
    sub.add_parser("testshift", help="recompute test token-change features with train statistics and rescore stages 1-3")
    sfin = sub.add_parser("final", help="stacker, generator rules, decision, same-address recovery; write and validate")
    sfin.add_argument("--shape", action="store_true", help="add the S1 score-shape features to the stacker")
    sfin.add_argument("--card", action="store_true", help="cardinality prior in the final decision (training countries)")
    sfin.add_argument("--card-all", action="store_true", help="with --card: apply the prior to every country (countries absent from training too)")
    sfin.add_argument("--gate", default="full", choices=["full", "narrow", "open"],
                      help="street-number gate for countries without digit typos: full (as detected), narrow (release renumbered / one-digit-indel copies), open (keep only legal-form conflicts)")
    sfin.add_argument("--neural", action="store_true", help="add the neural G9 features (run `neural` first) to the template stacker")
    sn = sub.add_parser("neural", help="train the char-level pair model on folds A+B (GPU) and score holdout and test pairs")
    sn.add_argument("--score-only", action="store_true", help="reuse <work>/nn/model.pt and model_b.pt (shipped in weights/)")
    ss = sub.add_parser("submit", help="tune the decision rule on H, decide test matches, write and validate")
    ss.add_argument("--score", default="p1", choices=["p1", "p2"])
    ss.add_argument("--rule", default="threshold", choices=["threshold", "expected_f"])
    ss.add_argument("--catswap", action="store_true", help="reject category-word swaps at the same address")
    sub.add_parser("diagnose", help="holdout breakdowns, orphan stress test, test country report, error samples")
    sa = sub.add_parser("all", help="run the full pipeline end to end")
    sa.add_argument("--score", default="p2", choices=["p1", "p2"])
    sa.add_argument("--rule", default="threshold", choices=["threshold", "expected_f"])
    sa.add_argument("--catswap", action="store_true", help="reject category-word swaps at the same address")
    sa.add_argument("--final", action="store_true", help="after stage 2 run stage3, testshift and final (the submitted pipeline)")
    sa.add_argument("--shape", action="store_true", help="with --final: add the S1 score-shape features to the stacker")
    sa.add_argument("--card", action="store_true", help="with --final: cardinality prior in the final decision")
    sa.add_argument("--card-all", action="store_true", help="with --final --card: prior for every country")
    sa.add_argument("--gate", default="full", choices=["full", "narrow", "open"], help="with --final: street-number gate mode")
    sa.add_argument("--neural", action="store_true", help="with --final: train and use the neural G9 features")
    sk = sub.add_parser("package", help="build <team>_submission.zip")
    sk.add_argument("--team", required=True)
    args = p.parse_args(argv)
    if args.cmd == "prepare":
        from .prepare import prepare_split
        prepare_split(args.split, config.DATA_DIR, config.WORK_DIR, config.N_JOBS)
    elif args.cmd == "block":
        from .blocking import DEFAULT, build_candidates
        build_candidates(args.split, config.WORK_DIR, DEFAULT)
    elif args.cmd == "block-report":
        from .blocking import recall_report
        print(recall_report(config.WORK_DIR, config.DATA_DIR))
    elif args.cmd == "prerank":
        from .model import save
        from .prerank import CAP, FLOOR, apply_prerank, train_prerank
        booster = train_prerank(config.WORK_DIR, config.DATA_DIR)
        save(booster, config.WORK_DIR / "models" / "prerank.txt")
        for split in ("train", "test"):
            apply_prerank(split, config.WORK_DIR, booster, FLOOR, CAP)
    elif args.cmd == "features":
        from .features import build_features
        build_features(args.split, config.WORK_DIR, config.DATA_DIR, config.N_JOBS)
    elif args.cmd == "tokens":
        from .tokens import augment_features
        augment_features(config.WORK_DIR, args.split, config.N_JOBS)
    elif args.cmd == "stage1":
        from .features import MODEL_FEATS
        from .model import fit_fold_models, predict_stage
        models = fit_fold_models(config.WORK_DIR, "feats", MODEL_FEATS, name="m1")
        for split in ("train", "test"):
            predict_stage(config.WORK_DIR, split, "feats", MODEL_FEATS, models, "p1")
    elif args.cmd == "stage2":
        from .context import CTX_FEATS, build_stage2_tables
        from .features import MODEL_FEATS
        from .model import fit_fold_models, predict_stage
        for split in ("train", "test"):
            build_stage2_tables(split, config.WORK_DIR, config.N_JOBS)
        feats = MODEL_FEATS + CTX_FEATS
        models = fit_fold_models(config.WORK_DIR, "feats2", feats, name="m2")
        for split in ("train", "test"):
            predict_stage(config.WORK_DIR, split, "feats2", feats, models, "p2")
    elif args.cmd == "stage3":
        from .stage3 import run_stage3
        run_stage3(config.WORK_DIR, config.N_JOBS)
    elif args.cmd == "testshift":
        from .testshift import rescore_test
        rescore_test(config.WORK_DIR, config.TC_DIR, config.N_JOBS)
    elif args.cmd == "final":
        from .final import run_final
        raise SystemExit(run_final(config.WORK_DIR, config.TC_DIR, config.OUTPUT_DIR, config.DATA_DIR, config.N_JOBS, args.shape, args.card, args.neural,
                                   args.card_all, args.gate))
    elif args.cmd == "neural":
        from . import neural
        if not args.score_only or not (neural.nn_dir(config.WORK_DIR) / "H_pairs.parquet").exists():
            neural.prep(config.WORK_DIR)                 # deterministic; never touches model.pt
        for tag, epochs, seed in neural.MODELS:            # model A (2 epochs) and model B (4 epochs); G9 averages them
            if not args.score_only:
                neural.train(config.WORK_DIR, epochs, seed=seed, tag=tag)
            for split in ("H", "test"):
                neural.score(config.WORK_DIR, config.TC_DIR, split, tag=tag)
    elif args.cmd == "submit":
        import json
        from .decide import decide, holdout_tune
        from .model import load_table
        from .package import validate, write_outputs
        excl = {}
        if args.catswap:
            from .rules import catswap_pairs
            excl = {s: catswap_pairs(config.WORK_DIR, s) for s in ("train", "test")}
        decision = holdout_tune(config.WORK_DIR, config.DATA_DIR, args.score, args.rule, excl.get("train"))
        decision["catswap"] = bool(args.catswap)
        print(decision)
        (config.WORK_DIR / "decision.json").write_text(json.dumps(decision, indent=2))
        pairs = load_table(config.WORK_DIR, "test", args.score, ["s1_id", "tgt_id", args.score])
        if args.catswap:
            from .rules import exclude_scores
            pairs = exclude_scores(pairs, excl["test"], args.score)
        m, c = write_outputs(config.WORK_DIR, config.OUTPUT_DIR, decide(pairs, args.score, decision))
        raise SystemExit(validate(m, c, config.DATA_DIR))
    elif args.cmd == "diagnose":
        import json
        from .diagnostics import run_diagnostics
        decision = json.loads((config.WORK_DIR / "decision.json").read_text())
        run_diagnostics(config.WORK_DIR, config.DATA_DIR, decision)
    elif args.cmd == "all":
        steps = [["prepare", "--split", "train"], ["prepare", "--split", "test"],
                 ["block", "--split", "train"], ["block", "--split", "test"], ["prerank"],
                 ["features", "--split", "train"], ["features", "--split", "test"], ["stage1"]]
        if args.score == "p2" or args.final:
            steps.append(["stage2"])
        if args.final:
            steps += [["stage3"], ["testshift"]] + ([["neural"]] if args.neural else [])
        for step in steps:
            print(">>", " ".join(step), flush=True)
            main(step)
        if args.final:
            main(["final"] + (["--shape"] if args.shape else []) + (["--card"] if args.card else []) + (["--neural"] if args.neural else [])
                 + (["--card-all"] if args.card_all else []) + ["--gate", args.gate])
        else:
            main(["submit", "--score", args.score, "--rule", args.rule] + (["--catswap"] if args.catswap else []))
    elif args.cmd == "package":
        from .package import build_zip
        z = build_zip(config.ROOT, config.OUTPUT_DIR, config.ROOT / "Documentation_template.md", args.team)
        print(z)


if __name__ == "__main__":
    main()
