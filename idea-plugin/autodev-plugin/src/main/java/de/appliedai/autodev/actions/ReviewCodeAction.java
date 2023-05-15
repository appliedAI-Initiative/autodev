package de.appliedai.autodev.actions;

import java.io.IOException;

public class ReviewCodeAction extends GenerateHtmlResponseEditorAction {
    @Override
    protected String generateResponse(String code) throws IOException, InterruptedException {
        return client.callCodeFunction("review", code);
    }
}