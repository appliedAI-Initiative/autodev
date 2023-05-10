package de.appliedai.autodev;

import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.actionSystem.LangDataKeys;
import com.intellij.openapi.actionSystem.PlatformDataKeys;
import com.intellij.openapi.command.WriteCommandAction;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.ui.Messages;
import com.intellij.psi.PsiFile;

public class AddCommentsAction extends AnAction {
    private ServiceClient client = new ServiceClient();

    @Override
    public void actionPerformed(AnActionEvent e) {
        Editor editor = getEditor(e);
        String selectedText = editor.getSelectionModel().getSelectedText();

        String pasteText;
        try {
            pasteText = client.addComments(selectedText);
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

    public void update(AnActionEvent e) {
        Editor editor = getEditor(e);
        if (editor == null)
            e.getPresentation().setEnabled(false);
    }

    private Editor getEditor(AnActionEvent e) {
        PsiFile psiFile = e.getData(LangDataKeys.PSI_FILE);
        Editor editor = e.getData(PlatformDataKeys.EDITOR);
        if (editor.getSelectionModel().getSelectedText() == null) {
            return null;
        }
        if (psiFile == null || editor == null) {
            return null;
        }
        return editor;
    }
}