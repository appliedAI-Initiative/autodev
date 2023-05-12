package de.appliedai.autodev;

import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.ui.Messages;

public class PotentialProblemsAction extends EditorAction {
    private ServiceClient client = new ServiceClient();

    @Override
    public void actionPerformed(AnActionEvent e) {
        String selectedText = this.getSelectedText(e);

        try {
            String result = client.checkForPotentialProblems(selectedText);
            boolean isHtml = true;
            AutoDevToolWindowManager.addContent(result, e.getProject(),
                    "Problems found", isHtml);
        }
        catch (Throwable t) {
            Messages.showInfoMessage(e.getProject(), e.toString(), "AutoCode Error");
        }
    }
}