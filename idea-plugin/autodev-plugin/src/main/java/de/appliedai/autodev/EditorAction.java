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

public abstract class EditorAction extends AnAction {
    protected ServiceClient client = new ServiceClient();

    protected String getSelectedText(AnActionEvent e) {
        Editor editor = getEditor(e);
        if (editor == null)
            return null;
        return editor.getSelectionModel().getSelectedText();
    }

    public void update(AnActionEvent e) {
        Editor editor = getEditor(e);
        if (editor == null)
            e.getPresentation().setEnabled(false);
    }

    protected Editor getEditor(AnActionEvent e) {
        PsiFile psiFile = e.getData(LangDataKeys.PSI_FILE);
        Editor editor = e.getData(PlatformDataKeys.EDITOR);
        if (psiFile == null || editor == null) {
            return null;
        }
        return editor;
    }
}