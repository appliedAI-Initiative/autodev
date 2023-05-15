package de.appliedai.autodev.actions;

import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.command.WriteCommandAction;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.ui.Messages;

import java.io.IOException;

public abstract class ReplaceTextEditorAction extends EditorAction {
    @Override
    public void actionPerformed(AnActionEvent e) {
        Editor editor = getEditor(e);
        if (editor == null)
            return;
        String selectedText = editor.getSelectionModel().getSelectedText();

        String pasteText;
        try {
            pasteText = this.obtainReplacementText(selectedText);
        }
        catch (Throwable t) {
            Messages.showInfoMessage(e.getProject(), e.toString(), "AutoCode Error");
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