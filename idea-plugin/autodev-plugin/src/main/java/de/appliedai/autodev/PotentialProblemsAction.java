package de.appliedai.autodev;

import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.command.WriteCommandAction;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.ui.Messages;

public class PotentialProblemsAction extends EditorAction {
    private ServiceClient client = new ServiceClient();

    @Override
    public void actionPerformed(AnActionEvent e) {
        String selectedText = this.getSelectedText(e);

        try {
            String result = client.checkForPotentialProblems(selectedText);
            AutoDevToolWindowManager.addContent("Here's what I found:\n" + result, e.getProject(), "Problems found");
        }
        catch (Throwable t) {
            Messages.showInfoMessage(e.getProject(), e.toString(), "AutoCode Error");
        }
    }
}