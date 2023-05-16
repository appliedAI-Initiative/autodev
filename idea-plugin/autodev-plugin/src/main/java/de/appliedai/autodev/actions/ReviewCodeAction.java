package de.appliedai.autodev.actions;

import java.io.IOException;
import java.io.PipedInputStream;

public class ReviewCodeAction extends GenerateToolWindowResponseEditorActionStreamed {
    @Override
    protected PipedInputStream generateResponse(String code) throws IOException {
        return client.callCodeFunctionStreamed("review", code);
    }
}