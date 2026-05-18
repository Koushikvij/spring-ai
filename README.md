# spring-ai

A small Java demo that uses DJL and Ollama to embed documents, build a simple retriever, and answer user questions in an interactive REPL.

## Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using:
- DJL for embeddings and model loading
- Ollama as the LLM backend
- A simple in-memory retriever
- Maven for build and execution

## Requirements

- Java 11
- Maven
- `ollama` installed and running
- An Ollama model pulled locally (for example: `mistral`)

## Setup

1. Start Ollama in a separate terminal:

```powershell
ollama serve
```

2. Pull a model if you have not already:

```powershell
ollama pull mistral
```

## Build

```powershell
mvn clean compile
```

## Run

The project is configured with Maven Exec plugin in `pom.xml`.

```powershell
mvn clean compile exec:java
```

If you prefer a direct command, the application entry point is `com.koushik.Main`.

## Maven Dependencies

The following DJL dependencies are present or implied by the current `pom.xml` configuration:

- `ai.djl:api:${djl.version}`
- `ai.djl.pytorch:pytorch-native-cpu` (use the OS-specific classifier)
- `ai.djl.huggingface:tokenizers:${djl.version}`
- `ai.djl.huggingface:sentence-transformers:${djl.version}`

Additional dependencies in the project:

- `com.google.code.gson:gson:2.10.1`
- `org.apache.commons:commons-math3:3.6.1`
- `org.junit.jupiter:junit-jupiter-api:5.9.2`
- `org.junit.jupiter:junit-jupiter-engine:5.9.2`
- `junit:junit:4.13.2`

## Optional: ONNX instead of PyTorch

If you want to use ONNX instead of PyTorch, replace the PyTorch dependency with:

```xml
<dependency>
    <groupId>ai.djl.onnxruntime</groupId>
    <artifactId>onnxruntime-engine</artifactId>
    <version>${djl.version}</version>
</dependency>
```

Then update `EmbeddingService` to use:

```java
optEngine("OnnxRuntime")
```

## Quick sanity check

1. Start Ollama:

```powershell
ollama serve
```

2. Pull a model:

```powershell
ollama pull mistral
```

3. Build the project:

```powershell
mvn clean compile
```

4. Run the demo:

```powershell
mvn clean compile exec:java
```

If the build succeeds and you see the interactive prompt, the setup is correct.

## Optional: Use LangChain4j

If you want a higher-level Java retriever+LLM chain instead of the hand-rolled pipeline, consider `langchain4j`.

Example dependency:

```xml
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-ollama</artifactId>
    <version>0.2.2</version>
</dependency>
```

You can then build a chain using `OllamaChatModel`, `EmbeddingModel`, and `RetrievalAugmentedGeneration`.

## Notes

- The application currently uses `com.koushik.Main` as its entry point.
- The `.gitignore` file ignores the full `target/` directory.
