package com.tabnineCommon.inline

import com.intellij.codeInsight.CodeInsightActionHandler
import com.intellij.codeInsight.actions.BaseCodeInsightAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.DumbAware
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiFile
import com.tabnineCommon.capabilities.RenderingMode
import com.tabnineCommon.general.DependencyContainer

class ManualTriggerTabnineInlineCompletionAction :
        BaseCodeInsightAction(false),
        DumbAware,
    InlineCompletionAction {
    companion object {
        const val ACTION_ID = "ManualTriggerTabnineInlineCompletionAction"
    }

    private val handler = DependencyContainer.singletonOfInlineCompletionHandler()
    private val completionsEventSender = DependencyContainer.instanceOfCompletionsEventSender()

    override fun getHandler(): CodeInsightActionHandler {
        return CodeInsightActionHandler { _: Project?, editor: Editor, _: PsiFile? ->
            System.out.println("Manual trigger inline: send trigger")
            completionsEventSender.sendManualSuggestionTrigger(RenderingMode.INLINE)
            System.out.println("Manual trigger inline: get current")
            val lastShownCompletion = CompletionPreview.getCurrentCompletion(editor)

            System.out.println("Manual trigger inline: retrieve and show")
            handler.retrieveAndShowCompletion(
                    editor, editor.caretModel.offset, lastShownCompletion, "",
                    DefaultCompletionAdjustment(),
                    true
            )
        }
    }

    override fun isValidForLookup(): Boolean = true
}
