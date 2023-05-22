package de.appliedai.autodev;

import com.intellij.openapi.project.Project;
import com.intellij.openapi.wm.RegisterToolWindowTask;
import com.intellij.openapi.wm.ToolWindow;
import com.intellij.openapi.wm.ToolWindowAnchor;
import com.intellij.openapi.wm.ToolWindowManager;
import com.intellij.ui.content.Content;
import com.intellij.ui.content.ContentFactory;

import javax.swing.*;
import javax.swing.text.Element;
import javax.swing.text.html.HTMLDocument;
import javax.swing.text.html.HTMLEditorKit;
import java.awt.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PipedInputStream;

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
        ToolWindowContent toolWindowContent = addTab("", project, tabName, isHtml);

        new Thread(() -> {
            try(is) {
                BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(is));
                char[] buf = new char[1];
                int numCharsRead;
                while ((numCharsRead = bufferedReader.read(buf)) != -1) {
                    toolWindowContent.append(String.valueOf(buf));
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
        private final HTMLEditorKit editorKit;
        private String content;
        private Element bodyElement = null;

        private final boolean isHtml;

        public ToolWindowContent(String content, boolean isHtml) {
            this.isHtml = isHtml;
            this.content = content;
            contentPanel.setLayout(new BorderLayout(0, 20));
            final int border = 5;
            contentPanel.setBorder(BorderFactory.createEmptyBorder(border, border, border, border));
            editorPane = new JEditorPane();
            editorPane.setEditable(false);
            if (isHtml) {
                editorPane.setContentType("text/html");
                editorPane.setSize(100, Integer.MAX_VALUE);
                editorKit = new HTMLEditorKit();
                editorPane.setEditorKit(editorKit);
                var css = editorKit.getStyleSheet();
                css.addRule("body, pre { font-family: Consolas; font-size: 13pt}");
                css.addRule("body { padding: 6px;}");
                css.addRule("pre { margin-left: 15px;}");
                var doc = editorKit.createDefaultDocument();
                editorPane.setDocument(doc);
                editorPane.setText(content);
            }
            else {
                editorKit = null;
                Font font = new Font("Consolas", Font.PLAIN, 13);
                editorPane.setFont(font);
                editorPane.setText(content);
                editorPane.setMargin(new Insets(6, 6, 6, 6));
            }
            contentPanel.add(editorPane);
        }

        public JPanel getContentPanel() {
            return contentPanel;
        }

        public void append(String addedContent) {
            this.content += addedContent;
            var document = editorPane.getDocument();
            try {
                if (!isHtml) {
                    document.insertString(document.getLength(), addedContent, null);
                }
                else {
                    // replace entire body tag with new content (this at least works better than just
                    // calling editorPane.setText, which causes constant flickering)
                    HTMLDocument htmlDocument = (HTMLDocument) document;
                    if (this.bodyElement == null) {
                        bodyElement = getBodyElement(htmlDocument);
                    }
                    htmlDocument.setOuterHTML(bodyElement, "<body>" + this.content + "</body>");
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        private static Element getBodyElement(final HTMLDocument document) {
            final Element body = findBodyElement(document.getDefaultRootElement());
            if (body == null) {
                throw new IllegalArgumentException("Not found <body> tag in given document.");
            }
            return body;
        }

        private static Element findBodyElement(final Element element) {
            if (element.getName().equals("body")) {
                return element.getElement(0);
            }
            for (int i = 0; i < element.getElementCount(); i++) {
                final Element child = findBodyElement(element.getElement(i));
                if (child != null) {
                    return child;
                }
            }
            return null;
        }
    }
}
