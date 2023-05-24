package de.appliedai.autodev.actions.base;

import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.ui.Messages;
import de.appliedai.autodev.AutoDevToolWindowManager;
import de.appliedai.autodev.ServiceClient;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;

public abstract class GenerateToolWindowResponseEditorActionStreamed extends EditorAction {
    @Override
    public void actionPerformed(@NotNull AnActionEvent e) {
        String selectedText = this.getSelectedText(e);
        try {
            var response = this.generateResponse(selectedText);
            AutoDevToolWindowManager.addTabStreamed(response, e.getProject(), "Response");
        }
        catch (Throwable t) {
            t.printStackTrace();
            Messages.showInfoMessage(e.getProject(), t.toString(), "AutoDev Error");
        }
    }

    protected abstract ServiceClient.StreamedResponse generateResponse(String code) throws IOException;
}