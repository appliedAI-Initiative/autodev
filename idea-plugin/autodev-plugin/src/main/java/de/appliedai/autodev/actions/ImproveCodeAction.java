package de.appliedai.autodev.actions;

import java.io.IOException;

public class ImproveCodeAction extends ReplaceTextEditorAction {
    @Override
    public String obtainReplacementText(String inputText) throws IOException, InterruptedException {
        return client.callCodeFunction("review", inputText);
    }
}