package de.appliedai.autodev.actions;

import java.io.IOException;

public class ExplainCodeAction extends GenerateHtmlResponseEditorAction {
    @Override
    protected String generateResponse(String code) throws IOException, InterruptedException {
        return client.callCodeFunction("explain", code);
    }
}