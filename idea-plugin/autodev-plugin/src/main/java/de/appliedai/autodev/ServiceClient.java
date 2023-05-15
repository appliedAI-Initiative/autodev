package de.appliedai.autodev;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class ServiceClient {
    private static String serviceUrl = "http://localhost:5000/";

    private HttpClient client;

    public ServiceClient() {
        this.client = HttpClient.newHttpClient();
    }

    private HttpRequest.BodyPublisher multipartData(Map<Object, Object> data, String boundary) throws IOException {
        // Result request body
        List<byte[]> byteArrays = new ArrayList<>();

        // Separator with boundary
        byte[] separator = ("--" + boundary + "\r\nContent-Disposition: form-data; name=").getBytes(StandardCharsets.UTF_8);

        // Iterating over data parts
        for (Map.Entry<Object, Object> entry : data.entrySet()) {

            // Opening boundary
            byteArrays.add(separator);

            // If value is type of Path (file) append content type with file name and file binaries, otherwise simply append key=value
            if (entry.getValue() instanceof Path) {
                Path path = (Path) entry.getValue();
                String mimeType = Files.probeContentType(path);
                byteArrays.add(("\"" + entry.getKey() + "\"; filename=\"" + path.getFileName()
                        + "\"\r\nContent-Type: " + mimeType + "\r\n\r\n").getBytes(StandardCharsets.UTF_8));
                byteArrays.add(Files.readAllBytes(path));
                byteArrays.add("\r\n".getBytes(StandardCharsets.UTF_8));
            } else {
                byteArrays.add(("\"" + entry.getKey() + "\"\r\n\r\n" + entry.getValue() + "\r\n")
                        .getBytes(StandardCharsets.UTF_8));
            }
        }

        // Closing boundary
        byteArrays.add(("--" + boundary + "--").getBytes(StandardCharsets.UTF_8));

        // Serializing as byte array
        return HttpRequest.BodyPublishers.ofByteArrays(byteArrays);
    }

    private String post(URI uri, HashMap<Object, Object> data) throws IOException, InterruptedException {
        String boundary = "autocode___";
        HttpRequest.BodyPublisher body = multipartData(data, boundary);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(uri)
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .POST(body)
                .build();
        HttpResponse<String> response = this.client.send(request, HttpResponse.BodyHandlers.ofString());
        return response.body();
    }

    public String addComments(String code) throws IOException, InterruptedException {
        return callCodeFunction("add-comments", code);
    }

    public String checkForPotentialProblems(String code) throws IOException, InterruptedException {
        return callCodeFunction("potential-problems", code);
    }

    public String callCodeFunction(String fn, String code) throws IOException, InterruptedException {
        HashMap<Object, Object> data = new HashMap<>();
        data.put("code", code);
        return post(URI.create(serviceUrl + "fn/" + fn), data);
    }
}
