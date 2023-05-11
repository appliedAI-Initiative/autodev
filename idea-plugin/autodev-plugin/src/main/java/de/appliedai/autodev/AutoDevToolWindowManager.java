package de.appliedai.autodev;

import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.RegisterToolWindowTask;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowAnchor;
import com.intellij.openapi.wm.ToolWindowManager;
import com.intellij.ui.content.Content;
import com.intellij.ui.content.ContentFactory;

import javax.swing.*;
import java.awt.*;

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

    public static void addContent(String s, Project project, String tabName) {
        ToolWindow toolWindow = getOrCreateToolWindow(project);
        ToolWindowContent toolWindowContent = new ToolWindowContent(s);
        Content content = ContentFactory.getInstance().createContent(toolWindowContent.getContentPanel(), "", false);
        if (tabName != null) {
            content.setTabName(tabName);
        }
        toolWindow.getContentManager().addContent(content);
        toolWindow.show();
    }

    private static class ToolWindowContent {
        private final JPanel contentPanel = new JPanel();

        public ToolWindowContent(String content) {
            contentPanel.setLayout(new BorderLayout(0, 20));
            final int border = 5;
            contentPanel.setBorder(BorderFactory.createEmptyBorder(border, border, border, border));
            JEditorPane editorPane = new JEditorPane();
            editorPane.setContentType("text/html");
            editorPane.setText("<pre>" + content + "</pre>");
            contentPanel.add(editorPane);
        }

        public JPanel getContentPanel() {
            return contentPanel;
        }
    }
}
