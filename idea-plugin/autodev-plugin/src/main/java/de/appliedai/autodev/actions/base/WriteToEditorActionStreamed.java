package de.appliedai.autodev.actions.base;

import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.command.WriteCommandAction;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.ui.Messages;
import de.appliedai.autodev.ServiceClient;
import org.jetbrains.annotations.NotNull;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public abstract class WriteToEditorActionStreamed extends EditorAction {
    @Override
    public void actionPerformed(@NotNull AnActionEvent e) {
        Editor editor = getEditor(e);
        String selectedText = this.getSelectedText(e);
        try {
            final var response = this.generateResponse(selectedText);
            Document editorDocument = editor.getDocument();
            new Thread(() -> {
                new WriteCommandAction.Simple(e.getProject()) {
                    @Override
                    protected void run() throws Throwable {
                        int start = editor.getSelectionModel().getSelectionStart();
                        int end = editor.getSelectionModel().getSelectionEnd();
                        editorDocument.replaceString(start, end, "");
                    }
                }.execute();
                try (response) {
                    BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(response.getInputStream()));
                    char[] buf = new char[1];
                    int numCharsRead;
                    while ((numCharsRead = bufferedReader.read(buf)) != -1) {
                        final int numReadCurrent = numCharsRead;
                        new WriteCommandAction.Simple(e.getProject()) {
                            @Override
                            protected void run() throws Throwable {
                                int pos = editor.getCaretModel().getCurrentCaret().getOffset();
                                editorDocument.insertString(pos, String.valueOf(buf));
                                editor.getCaretModel().moveToOffset(pos + numReadCurrent);
                            }
                        }.execute();
                    }
                } catch (Throwable t) {
                    t.printStackTrace();
                }
            }).start();
        }
        catch (Throwable t) {
            t.printStackTrace();
            Messages.showInfoMessage(e.getProject(), t.toString(), "AutoDev Error");
        }
    }

    protected abstract ServiceClient.StreamedResponse generateResponse(String code) throws IOException;
}