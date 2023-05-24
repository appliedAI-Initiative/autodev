package de.appliedai.autodev.actions;

import de.appliedai.autodev.ServiceClient;
import de.appliedai.autodev.actions.base.GenerateToolWindowResponseEditorActionStreamed;

import java.io.IOException;

public class ReviewCodeAction extends GenerateToolWindowResponseEditorActionStreamed {
    @Override
    protected ServiceClient.StreamedResponse generateResponse(String code) throws IOException {
        return client.callCodeFunctionStreamed("review", code);
    }
}