package com.tabnineCommon.inline;

import com.intellij.codeInsight.hint.HintManager;
import com.intellij.codeInsight.hint.HintManagerImpl;
import com.intellij.codeInsight.hint.HintUtil;
import com.intellij.openapi.actionSystem.ActionManager;
import com.intellij.openapi.diagnostic.Logger;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.keymap.KeymapUtil;
import com.intellij.openapi.util.IconLoader;
import com.intellij.openapi.util.text.StringUtil;
import com.intellij.ui.LightweightHint;
import com.intellij.ui.SimpleColoredComponent;
import com.intellij.ui.SimpleColoredText;
import com.intellij.ui.SimpleTextAttributes;
import de.appliedai.autodev.AutoDevConfig;
import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;


public class InlineKeybindingHintUtil {
  private static Icon textLogo = IconLoader.findIcon("/autodev/text-logo.png",
          AutoDevConfig.class.getClassLoader());

  public static void createAndShowHint(@NotNull Editor editor, @NotNull Point pos) {
    try {
      HintManagerImpl.getInstanceImpl()
          .showEditorHint(
              new LightweightHint(createInlineHintComponent()),
              editor,
              pos,
              HintManager.HIDE_BY_ANY_KEY | HintManager.UPDATE_BY_SCROLLING,
              0,
              false);
    } catch (Throwable e) {
      Logger.getInstance(InlineKeybindingHintUtil.class)
          .warn("Failed to show inline key bindings hints", e);
    }
  }

  private static JComponent createInlineHintComponent() {
    SimpleColoredComponent component = HintUtil.createInformationComponent();

    component.setIconOnTheRight(true);
    component.setIcon(textLogo);

    SimpleColoredText coloredText =
        new SimpleColoredText(hintText(), SimpleTextAttributes.REGULAR_ATTRIBUTES);

    coloredText.appendToComponent(component);

    return new InlineKeybindingHintComponent(component);
  }

  private static String hintText() {
    String nextShortcut = getShortcutText(ShowNextTabnineInlineCompletionAction.ACTION_ID);
    String prevShortcut = getShortcutText(ShowPreviousTabnineInlineCompletionAction.ACTION_ID);
    String acceptShortcut = getShortcutText(AcceptTabnineInlineCompletionAction.ACTION_ID);
    String cancelShortcut = KeymapUtil.getKeyText(KeyEvent.VK_ESCAPE);

    return String.format(
        "Next (%s) Prev (%s) Accept (%s) Cancel (%s)",
        nextShortcut, prevShortcut, acceptShortcut, cancelShortcut);
  }

  private static String getShortcutText(String actionId) {
    return StringUtil.defaultIfEmpty(
        KeymapUtil.getFirstKeyboardShortcutText(ActionManager.getInstance().getAction(actionId)),
        "Missing shortcut key");
  }
}
