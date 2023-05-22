package de.appliedai.autodev.actions.base;

import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.ui.Messages;
import de.appliedai.autodev.AutoDevToolWindowManager;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.io.PipedInputStream;

public abstract class GenerateToolWindowResponseEditorActionStreamed extends EditorAction {
    private final boolean isHtml;

    public GenerateToolWindowResponseEditorActionStreamed(boolean isHtml) {
        this.isHtml = isHtml;
    }

    @Override
    public void actionPerformed(@NotNull AnActionEvent e) {
        String selectedText = this.getSelectedText(e);
        try {
            PipedInputStream is = this.generateResponse(selectedText);
            AutoDevToolWindowManager.addTabStreamed(is, e.getProject(),
                    "Response", isHtml);
        }
        catch (Throwable t) {
            t.printStackTrace();
            Messages.showInfoMessage(e.getProject(), t.toString(), "AutoDev Error");
        }
    }

    protected abstract PipedInputStream generateResponse(String code) throws IOException;
}