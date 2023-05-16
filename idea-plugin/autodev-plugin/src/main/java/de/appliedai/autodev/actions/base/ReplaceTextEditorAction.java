package de.appliedai.autodev.actions.base;

import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.command.WriteCommandAction;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.ui.Messages;
import de.appliedai.autodev.actions.base.EditorAction;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;

public abstract class ReplaceTextEditorAction extends EditorAction {
    @Override
    public void actionPerformed(@NotNull AnActionEvent e) {
        Editor editor = getEditor(e);
        if (editor == null)
            return;
        String selectedText = editor.getSelectionModel().getSelectedText();

        String pasteText;
        try {
            pasteText = this.obtainReplacementText(selectedText);
        }
        catch (Throwable t) {
            t.printStackTrace();
            Messages.showInfoMessage(e.getProject(), t.toString(), "AutoDev Error");
            return;
        }

        Document editorDocument = editor.getDocument();
        String finalPasteText = pasteText;
        new WriteCommandAction.Simple(e.getProject()) {
            @Override
            protected void run() throws Throwable {
                int start = editor.getSelectionModel().getSelectionStart();
                int end = editor.getSelectionModel().getSelectionEnd();
                editorDocument.replaceString(start, end, finalPasteText);
            }
        }.execute();
    }

    public abstract String obtainReplacementText(String inputText) throws IOException, InterruptedException;
}