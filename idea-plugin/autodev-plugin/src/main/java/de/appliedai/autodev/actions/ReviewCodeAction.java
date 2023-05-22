package de.appliedai.autodev.actions;

import de.appliedai.autodev.actions.base.GenerateToolWindowResponseEditorActionStreamed;

import java.io.IOException;
import java.io.PipedInputStream;

public class ReviewCodeAction extends GenerateToolWindowResponseEditorActionStreamed {
    public ReviewCodeAction() {
        super(true);
    }

    @Override
    protected PipedInputStream generateResponse(String code) throws IOException {
        return client.callCodeFunctionStreamed("review", code);
    }
}