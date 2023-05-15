package de.appliedai.autodev.actions;

import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.ui.Messages;
import de.appliedai.autodev.AutoDevToolWindowManager;

import java.io.IOException;

public abstract class GenerateHtmlResponseEditorAction extends EditorAction {
    @Override
    public void actionPerformed(AnActionEvent e) {
        String selectedText = this.getSelectedText(e);
        try {
            String result = client.checkForPotentialProblems(selectedText);
            boolean isHtml = true;
            AutoDevToolWindowManager.addContent(result, e.getProject(),
                    "Response", isHtml);
        }
        catch (Throwable t) {
            Messages.showInfoMessage(e.getProject(), e.toString(), "AutoCode Error");
        }
    }

    protected abstract String generateResponse(String code) throws IOException, InterruptedException;
}