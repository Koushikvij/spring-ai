package com.koushik;

import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

/**
 * Run it with:
 *
 *   mvn compile exec:java -Dexec.mainClass="com.example.rag.Main"
 *
 * Make sure you have started Ollama (`ollama serve`) and pulled the model you want
 * (e.g. `ollama pull mistral`).
 */
public class Main {

    public static void main(String[] args) throws Exception {
        // ---------- 1️⃣ Sample knowledge base ----------
        List<String> docs = Arrays.asList(
                "The Eiffel Tower was completed in 1889 and stands in Paris, France.",
                "The Great Barrier Reef is the world's largest coral reef system, located off the coast of Queensland, Australia.",
                "Python is an interpreted, high‑level programming language created by Guido van Rossum and first released in 1991.",
                "The telephone was invented by Alexander Graham Bell in 1876.",
                "Mount Everest is the highest mountain on Earth, with a summit elevation of 8,848 m above sea level.",
                "The Mona Lisa is a portrait painting by Leonardo da Vinci, created in the early 16th century.",
                "The Amazon rainforest is the largest tropical rainforest in the world, covering much of South America",
                "The human brain contains approximately 86 billion neurons.",
                "The first successful powered airplane flight was made by the Wright brothers in 1903.",
                "The Great Wall of China is a series of fortifications built across the historical northern borders of China.",
                "Albert Einstein developed the theory of relativity, one of the two pillars of modern physics, in the early 20th century.",
                "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
                "Java is a high‑level, class‑based, object‑oriented programming language that is designed to have as few implementation dependencies as possible.",
                "The Taj Mahal is an ivory‑white marble mausoleum on the right bank of the Yamuna river in the Indian city of Agra.",
                "The human heart has four chambers: the left atrium, left ventricle, right atrium, and right ventricle."
        );

        // ---------- 2️⃣ Load embedding model ----------
        System.out.println("Loading embedding model (e5‑small‑v2) …");
        EmbeddingService embeddingService = new EmbeddingService("intfloat/e5-small-v2");

        // ---------- 3️⃣ Build in‑memory retriever ----------
        System.out.println("Embedding knowledge base …");
        Retriever retriever = new Retriever(docs, embeddingService);

        // ---------- 4️⃣ Prepare Ollama client ----------
        // Change the model name if you pulled a different one.
        OllamaChatClient chatClient = new OllamaChatClient("mistral");

        // ---------- 5️⃣ Assemble the RAG agent ----------
        RagAgent agent = new RagAgent(retriever, embeddingService, chatClient, 3); // top‑3 docs

        // ---------- 6️⃣ Interactive loop ----------
        Scanner scanner = new Scanner(System.in);
        System.out.println("\n🔎 Enter a question (type ‘exit’ to quit):");
        while (true) {
            System.out.print("\n> ");
            String question = scanner.nextLine().trim();
            if (question.equalsIgnoreCase("exit") || question.isEmpty())
                break;

            System.out.println("\n⏳ Thinking …");
            try {
                String answer = agent.answer(question);
                System.out.println("\n💬 Answer:\n" + answer);
            } catch (Exception e) {
                System.err.println("❌ Something went wrong: " + e.getMessage());
                e.printStackTrace();
            }
        }

        // Clean up resources
        embeddingService.close();
        System.out.println("\n👋 Bye!");
    }
}