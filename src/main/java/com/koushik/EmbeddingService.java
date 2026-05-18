package com.koushik;

import ai.djl.Application;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.nn.core.Embedding;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.embedding.TextEmbedding;
import ai.djl.modality.nlp.embedding.WordEmbedding;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.modality.nlp.preprocess.Tokenizer;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslatorFactory;
import ai.djl.translate.TranslatorFactory;


import java.io.IOException;
import java.util.List;

/**
 * Wraps a HuggingFace sentence‑transformer model (e5‑small‑v2,
 * all‑MiniLM‑L6‑v2, etc.) and returns L2‑normalized float vectors.
 *
 * DJL’s {@code SentenceEmbedding} pipeline does the heavy lifting.
 *
 * The model is loaded only once; {@link #embed(String)} can be called many times
 * (it is thread‑safe – DJL’s {@code Predictor} is synchronized internally).
 */
public class EmbeddingService implements AutoCloseable {

    private final ZooModel<String, float[]> model;
    private final ai.djl.inference.Predictor<String, float[]> predictor;

    /**
     * Construct an EmbeddingService for the given HuggingFace model id.
     *
     * @param modelId the HF identifier, e.g. "intfloat/e5-small-v2"
     * @throws IOException   if the model cannot be downloaded
     * @throws ModelException if the model cannot be loaded
     */
    public EmbeddingService(String modelId) throws IOException, ModelException {
        // Build a criteria that tells DJL what we want:
        Criteria<String, float[]> criteria = Criteria.builder()
                .optApplication(Application.NLP.TEXT_EMBEDDING)
                .setTypes(String.class, float[].class)
                .optModelUrls("djl://ai.djl.huggingface.pytorch/" + modelId)
                // Force the use of the PyTorch engine (or ONNX if you prefer).
                .optEngine("PyTorch")
                .optProgress(new ai.djl.training.util.ProgressBar())
                .build();

        this.model = criteria.loadModel();
        this.predictor = model.newPredictor();
    }

    /**
     * Compute a **L2‑normalized** embedding for a single piece of text.
     *
     * @param text the raw string
     * @return a float[] of length (usually) 384 for e5‑small‑v2
     * @throws TranslateException if inference fails
     */
    public float[] embed(String text) throws TranslateException {
        float[] raw = predictor.predict(text);
        // L2‑normalize to make cosine similarity = dot‑product
        double norm = 0.0;
        for (float v : raw) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);
        if (norm == 0) {
            return raw; // avoid division by zero
        }
        float[] normed = new float[raw.length];
        for (int i = 0; i < raw.length; i++) {
            normed[i] = (float) (raw[i] / norm);
        }
        return normed;
    }

    @Override
    public void close() {
        predictor.close();
        model.close();
    }
}