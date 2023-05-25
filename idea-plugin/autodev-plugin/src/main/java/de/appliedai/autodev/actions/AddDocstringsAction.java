package de.appliedai.autodev.actions;

import de.appliedai.autodev.ServiceClient;
import de.appliedai.autodev.actions.base.WriteToEditorActionStreamed;

import java.io.IOException;

public class AddDocstringsAction extends WriteToEditorActionStreamed {
    @Override
    protected ServiceClient.StreamedResponse generateResponse(String code) throws IOException {
        return client.callCodeFunctionStreamed("add-docstrings", code);
    }
}