package de.appliedai.autodev.actions;

import java.io.IOException;

public class ImplementTestsAction extends GenerateHtmlResponseEditorAction {
    @Override
    protected String generateResponse(String code) throws IOException, InterruptedException {
        return client.callCodeFunction("implement-tests", code);
    }
}