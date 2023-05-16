package de.appliedai.autodev.actions;

import java.io.IOException;

public class AddDocstringsAction extends ReplaceTextEditorAction {
    @Override
    public String obtainReplacementText(String inputText) throws IOException, InterruptedException {
        return client.callCodeFunction("add-comments", inputText);
    }
}