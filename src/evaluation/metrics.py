def recall_at_k(chunks_info, relevant_chunk_ids, k):
    """
    chunks_info: list of dicts (from get_answer)
    relevant_chunk_ids: list[int]
    k: int

    Returns: float (0 or 1 for binary recall)
    """
    if not chunks_info:
        return 0.0

    top_k = chunks_info[:k]
    retrieved_ids = {c["chunk_id"] for c in top_k}

    # Binary recall: did we retrieve ANY relevant chunk?
    return float(any(cid in retrieved_ids for cid in relevant_chunk_ids))