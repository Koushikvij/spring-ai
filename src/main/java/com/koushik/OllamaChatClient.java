package com.koushik;

import com.google.gson.*;

import java.io.IOException;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * Thin wrapper around Ollama's OpenAI‑compatible chat API.
 *
 * Ollama runs locally at http://127.0.0.1:11434 . The client sends a JSON payload,
 * receives a streaming response (optional) and returns the final answer.
 *
 * You can change the model name (default "mistral") to any model you have
 * pulled with `ollama pull <model>`.
 */
public class OllamaChatClient {

    private static final String DEFAULT_BASE_URL = "http://127.0.0.1:11434";
    private final HttpClient httpClient;
    private final Gson gson;
    private final String baseUrl;
    private final String modelName;

    /**
     * @param modelName the model you want to talk to, e.g. "mistral", "mixtral", "gemma"
     */
    public OllamaChatClient(String modelName) {
        this(DEFAULT_BASE_URL, modelName);
    }

    public OllamaChatClient(String baseUrl, String modelName) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.modelName = modelName;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
        this.gson = new Gson();
    }

    /**
     * Sends a single‑turn prompt to Ollama and returns the model's answer.
     *
     * @param prompt the full prompt (including the retrieved context)
     * @return the generated answer (trimmed)
     * @throws IOException          if the HTTP request fails
     * @throws InterruptedException if the thread is interrupted
     */
    public String chat(String prompt) throws IOException, InterruptedException {
        // Build request body
        JsonObject body = new JsonObject();
        body.addProperty("model", modelName);
        JsonArray messages = new JsonArray();

        JsonObject userMessage = new JsonObject();
        userMessage.addProperty("role", "user");
        userMessage.addProperty("content", prompt);
        messages.add(userMessage);

        body.add("messages", messages);
        body.addProperty("temperature", 0.7);
        body.addProperty("max_tokens", 512);
        body.addProperty("stream", false); // set true for streaming (not needed for simple demo)

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(body)))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new IOException("Ollama API returned status " + response.statusCode() + ": " + response.body());
        }

        // Parse the OpenAI‑compatible response
        JsonObject json = JsonParser.parseString(response.body()).getAsJsonObject();
        JsonArray choices = json.getAsJsonArray("choices");
        if (choices.size() == 0) {
            throw new IOException("No choices returned from Ollama");
        }
        JsonObject firstChoice = choices.get(0).getAsJsonObject();
        JsonObject message = firstChoice.getAsJsonObject("message");
        String content = message.get("content").getAsString();
        return content.trim();
    }
}