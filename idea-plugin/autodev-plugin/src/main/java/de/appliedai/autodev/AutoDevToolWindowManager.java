package de.appliedai.autodev;

import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.RegisterToolWindowTask;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowAnchor;
import com.intellij.openapi.wm.ToolWindowManager;
import com.intellij.ui.content.Content;
import com.intellij.ui.content.ContentFactory;

import javax.swing.*;
import javax.swing.text.BadLocationException;
import java.awt.*;
import java.io.*;

public class AutoDevToolWindowManager {
    private static final String toolWindowId = "AutoDev";

    public static ToolWindow getOrCreateToolWindow(Project project) {
        ToolWindow toolWindow = ToolWindowManager.getInstance(project).getToolWindow(toolWindowId);
        if(toolWindow == null) {
            var task = new RegisterToolWindowTask(toolWindowId, ToolWindowAnchor.RIGHT, null, false, false, true, true, null, null, null);
            toolWindow = ToolWindowManager.getInstance(project).registerToolWindow(task);
        }
        return toolWindow;
    }

    public static ToolWindowContent addTab(String s, Project project, String tabName, boolean isHtml) {
        ToolWindow toolWindow = getOrCreateToolWindow(project);
        ToolWindowContent toolWindowContent = new ToolWindowContent(s, isHtml);
        Content content = ContentFactory.getInstance().createContent(toolWindowContent.getContentPanel(), tabName, false);
        if (tabName != null) {
            content.setTabName(tabName);
        }
        toolWindow.getContentManager().addContent(content);
        toolWindow.getContentManager().setSelectedContent(content);
        toolWindow.show();
        return toolWindowContent;
    }

    public static void addTabStreamed(PipedInputStream is, Project project, String tabName, boolean isHtml) throws IOException {
        ToolWindowContent toolWindowContent = addTab("", project, tabName, false);

        new Thread(() -> {
            try(is) {
                BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(is));
                String line;
                while ((line = bufferedReader.readLine()) != null) {
                    try {
                        toolWindowContent.append(line + "\n");
                    }
                    catch (BadLocationException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
            catch (IOException e) {
                e.printStackTrace();
            }
        }).start();
    }

    private static class ToolWindowContent {
        private final JPanel contentPanel = new JPanel();
        private final JEditorPane editorPane;

        public ToolWindowContent(String content, boolean isHtml) {
            contentPanel.setLayout(new BorderLayout(0, 20));
            final int border = 5;
            contentPanel.setBorder(BorderFactory.createEmptyBorder(border, border, border, border));
            editorPane = new JEditorPane();
            if (isHtml) {
                editorPane.setContentType("text/html");
            }
            else {
                Font font = new Font("Consolas", Font.PLAIN, 13);
                editorPane.setFont(font);
            }
            editorPane.setText(content);
            contentPanel.add(editorPane);
        }

        public JPanel getContentPanel() {
            return contentPanel;
        }

        public void append(String content) throws BadLocationException {
            var document = editorPane.getDocument();
            document.insertString(document.getLength(), content, null);
        }
    }
}
