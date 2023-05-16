package de.appliedai.autodev.actions;

import de.appliedai.autodev.actions.base.GenerateToolWindowResponseEditorActionStreamed;

import java.io.IOException;
import java.io.PipedInputStream;

public class PotentialProblemsAction extends GenerateToolWindowResponseEditorActionStreamed {
    @Override
    protected PipedInputStream generateResponse(String code) throws IOException {
        return client.callCodeFunctionStreamed("potential-problems", code);
    }
}