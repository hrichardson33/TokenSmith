import json
from src.main import get_answer
from src.config import RAGConfig
from src.instrumentation.logging import get_logger
from src.retriever import load_artifacts
from src.ranking.ranker import EnsembleRanker
from src.retriever import FAISSRetriever, BM25Retriever

from src.evaluation.metrics import recall_at_k


def load_system(cfg, args):
    artifacts_dir = cfg.get_artifacts_directory()

    faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(
        artifacts_dir, args.index_prefix
    )

    retrievers = [
        FAISSRetriever(faiss_idx, cfg.embed_model),
        BM25Retriever(bm25_idx),
    ]

    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )

    return {
        "chunks": chunks,
        "sources": sources,
        "retrievers": retrievers,
        "ranker": ranker,
        "meta": meta,
    }


def run_eval(cfg, args):
    with open("tests/ground_truth.json") as f:
        ground_truth = json.load(f)

    artifacts = load_system(cfg, args)
    logger = get_logger()

    results = []

    for query, relevant_ids in ground_truth.items():
        print(f"\nEvaluating: {query}")

        _, chunks_info, _ = get_answer(
            query,
            cfg,
            args,
            logger=logger,
            console=None,
            artifacts=artifacts,
            is_test_mode=True
        )

        r1 = recall_at_k(chunks_info, relevant_ids, 1)
        r3 = recall_at_k(chunks_info, relevant_ids, 3)
        r5 = recall_at_k(chunks_info, relevant_ids, 5)

        results.append((r1, r3, r5))

        print(f"Recall@1={r1}, Recall@3={r3}, Recall@5={r5}")

    return results


def summarize(results):
    n = len(results)

    avg_r1 = sum(r[0] for r in results) / n
    avg_r3 = sum(r[1] for r in results) / n
    avg_r5 = sum(r[2] for r in results) / n

    print("\n=== FINAL RESULTS ===")
    print(f"Recall@1: {avg_r1:.3f}")
    print(f"Recall@3: {avg_r3:.3f}")
    print(f"Recall@5: {avg_r5:.3f}")


if __name__ == "__main__":
    from src.main import parse_args
    import pathlib

    args = parse_args()
    cfg = RAGConfig.from_yaml(pathlib.Path("config/config.yaml"))

    results = run_eval(cfg, args)
    summarize(results)