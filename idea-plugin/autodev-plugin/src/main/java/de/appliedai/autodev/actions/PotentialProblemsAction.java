package de.appliedai.autodev.actions;

import java.io.IOException;

public class PotentialProblemsAction extends GenerateHtmlResponseEditorAction {
    @Override
    protected String generateResponse(String code) throws IOException, InterruptedException {
        return client.checkForPotentialProblems(code);
    }
}