package de.appliedai.autodev.actions.base;

import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.ui.Messages;
import de.appliedai.autodev.AutoDevToolWindowManager;
import de.appliedai.autodev.actions.base.EditorAction;

import java.io.IOException;

public abstract class GenerateToolWindowResponseEditorAction extends EditorAction {
    @Override
    public void actionPerformed(AnActionEvent e) {
        String selectedText = this.getSelectedText(e);
        try {
            String result = this.generateResponse(selectedText);
            boolean isHtml = true;
            AutoDevToolWindowManager.addTab(result, e.getProject(),
                    "Response", isHtml);
        }
        catch (Throwable t) {
            t.printStackTrace();
            Messages.showInfoMessage(e.getProject(), t.toString(), "AutoDev Error");
        }
    }

    protected abstract String generateResponse(String code) throws IOException, InterruptedException;
}