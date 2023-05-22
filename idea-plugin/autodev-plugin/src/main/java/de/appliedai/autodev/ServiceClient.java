package de.appliedai.autodev;

import org.apache.http.HttpEntity;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

import java.io.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

    /**
     * @param uri the URI of the function to be called
     * @param data the data to be posted
     * @param outputStream the stream to which the response shall be written
     * @throws IOException
     */
    private void post(URI uri, HashMap<String, String> data, PipedOutputStream outputStream) throws IOException {
        try(CloseableHttpClient httpClient = HttpClients.createDefault()) {
            HttpPost httpPost = new HttpPost(uri);

            MultipartEntityBuilder builder = MultipartEntityBuilder.create();
            data.forEach(builder::addTextBody);

            HttpEntity entity = builder.build();
            httpPost.setEntity(entity);

            org.apache.http.HttpResponse response = httpClient.execute(httpPost);
            System.out.println("Response status: " + response.getStatusLine());
            InputStream inputStream = response.getEntity().getContent();
            try (outputStream) {
                byte[] buffer = new byte[2];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
            }
            finally {
                EntityUtils.consume(response.getEntity());
            }
        }
    }

    public String callCodeFunction(String fn, String code) throws IOException, InterruptedException {
        HashMap<Object, Object> data = new HashMap<>();
        data.put("code", code);
        return post(URI.create(serviceUrl + "fn/" + fn), data);
    }

    public PipedInputStream callCodeFunctionStreamed(String fn, String code) throws IOException {
        HashMap<String, String> data = new HashMap<>();
        data.put("code", code);

        PipedInputStream is = new PipedInputStream();
        @SuppressWarnings("resource") PipedOutputStream os = new PipedOutputStream();
        os.connect(is);

        new Thread(() -> {
            try {
                post(URI.create(serviceUrl + "fn/stream/" + fn), data, os);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }).start();

        return is;
    }
}
