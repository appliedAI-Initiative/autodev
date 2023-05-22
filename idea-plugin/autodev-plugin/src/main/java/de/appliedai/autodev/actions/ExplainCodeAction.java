package de.appliedai.autodev.actions;

import de.appliedai.autodev.actions.base.GenerateToolWindowResponseEditorActionStreamed;

import java.io.IOException;
import java.io.PipedInputStream;

public class ExplainCodeAction extends GenerateToolWindowResponseEditorActionStreamed {
    public ExplainCodeAction() {
        super(true);
    }

    @Override
    protected PipedInputStream generateResponse(String code) throws IOException {
        return client.callCodeFunctionStreamed("explain", code);
    }
}