package com.koushik;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Keeps a list of documents and their pre‑computed embeddings.
 * Performs a simple top‑k cosine‑similarity search.
 */
public class Retriever {

    private final List<String> documents;
    private final List<float[]> embeddings;
    private final int dimension;

    /**
     * @param docs   raw text documents (any size you want)
     * @param embedFn a function that turns a String into a normalized float[]
     */
    public Retriever(List<String> docs, EmbeddingService embeddingService) throws Exception {
        this.documents = new ArrayList<>(docs);
        this.embeddings = new ArrayList<>(docs.size());
        for (String doc : docs) {
            embeddings.add(embeddingService.embed(doc));
        }
        if (!embeddings.isEmpty()) {
            this.dimension = embeddings.get(0).length;
        } else {
            this.dimension = 0;
        }
    }

    /**
     * Return the top‑k most similar documents for a query string.
     *
     * @param query  the user question
     * @param k      number of candidates to return
     * @param embeddingService used to embed the query (we reuse the same service)
     * @return a list of documents, ordered from most to least similar
     */
    public List<String> retrieve(String query, int k, EmbeddingService embeddingService) throws Exception {
        float[] queryVec = embeddingService.embed(query);
        // Use a priority queue (min‑heap) to keep best‑k
        PriorityQueue<ScoredDoc> heap = new PriorityQueue<>(Comparator.comparingDouble(d -> d.score));

        for (int i = 0; i < embeddings.size(); i++) {
            float[] docVec = embeddings.get(i);
            double score = dot(queryVec, docVec);   // because vectors are L2‑normed, dot == cosine
            if (heap.size() < k) {
                heap.offer(new ScoredDoc(i, score));
            } else if (score > heap.peek().score) {
                heap.poll();
                heap.offer(new ScoredDoc(i, score));
            }
        }

        // Pull items out of heap, sort descending
        List<ScoredDoc> top = new ArrayList<>(heap);
        top.sort((a, b) -> Double.compare(b.score, a.score));

        return top.stream()
                .map(sd -> documents.get(sd.docId))
                .collect(Collectors.toList());
    }

    /** Simple dot product (vectors are already normalized). */
    private static double dot(float[] a, float[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /** Helper class to keep doc index + similarity score. */
    private static class ScoredDoc {
        final int docId;
        final double score;
        ScoredDoc(int docId, double score) {
            this.docId = docId;
            this.score = score;
        }
    }
}