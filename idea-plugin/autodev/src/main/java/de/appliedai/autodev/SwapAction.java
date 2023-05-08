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

import java.awt.Toolkit;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.StringSelection;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.io.IOException;

/*
 Example action from https://github.com/sajjad-ahmed/Swap-IntelliJ-Plugin
 */
public class SwapAction extends AnAction
{
    @Override
    public void actionPerformed(AnActionEvent e)
    {
        Editor editor = getEditor(e);
        String selectedText = editor.getSelectionModel().getSelectedText();

        Toolkit toolkit = Toolkit.getDefaultToolkit();
        Clipboard clipboard = toolkit.getSystemClipboard();
        String pasteText;
        try
        {
            pasteText = (String) clipboard.getData(DataFlavor.stringFlavor);
            StringSelection stringSelection = new StringSelection(selectedText);
            clipboard.setContents(stringSelection, stringSelection);
        } catch (UnsupportedFlavorException e1)
        {
            Messages.showInfoMessage(e.getProject(), "The clipboard doesn't contain any string.", "Can't Swap");
            return;
        } catch (IOException e1)
        {
            return;
        }
        Document editorDocument = editor.getDocument();
        String finalPasteText = pasteText;
        new WriteCommandAction.Simple(e.getProject())
        {
            @Override
            protected void run() throws Throwable
            {
                int start = editor.getSelectionModel().getSelectionStart();
                int end = editor.getSelectionModel().getSelectionEnd();
                editorDocument.replaceString(start, end, finalPasteText);
            }
        }.execute();
    }

    public void update(AnActionEvent e)
    {
        Editor editor = getEditor(e);
        if (editor == null)
            e.getPresentation().setEnabled(false);
    }

    private Editor getEditor(AnActionEvent e)
    {
        PsiFile psiFile = e.getData(LangDataKeys.PSI_FILE);
        Editor editor = e.getData(PlatformDataKeys.EDITOR);
        if (editor.getSelectionModel().getSelectedText() == null)
        {
            return null;
        }
        if (psiFile == null || editor == null)
        {
            return null;
        }
        return editor;
    }
}