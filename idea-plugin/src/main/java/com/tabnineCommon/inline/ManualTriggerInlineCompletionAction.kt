package com.tabnineCommon.inline

import com.intellij.codeInsight.CodeInsightActionHandler
import com.intellij.codeInsight.actions.BaseCodeInsightAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.project.DumbAware
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiFile
import com.tabnineCommon.capabilities.RenderingMode
import com.tabnineCommon.general.DependencyContainer
import de.appliedai.autodev.util.TaskLogger
import de.appliedai.autodev.util.TempLogger
import de.appliedai.autodev.autocomplete.InlineCompletionHandler

class ManualTriggerInlineCompletionAction :
        BaseCodeInsightAction(false),
        DumbAware,
    InlineCompletionAction {
    companion object {
        const val ACTION_ID = "ManualTriggerTabnineInlineCompletionAction"
    }

    private val log = TempLogger(ManualTriggerInlineCompletionAction::class.java)
    private var nextTaskId = 1
    private val handler = InlineCompletionHandler.getInstance()
    private val completionsEventSender = DependencyContainer.instanceOfCompletionsEventSender()

    override fun getHandler(): CodeInsightActionHandler {
        return CodeInsightActionHandler { _: Project?, editor: Editor, _: PsiFile? ->
            val log = TaskLogger(log, String.format("Manual%d - ", nextTaskId++))
            completionsEventSender.sendManualSuggestionTrigger(RenderingMode.INLINE)
            val lastShownCompletion = CompletionPreview.getCurrentCompletion(editor)
            handler.retrieveAndShowCompletion(
                    editor, editor.caretModel.offset, lastShownCompletion, "",
                    DefaultCompletionAdjustment(),
                    true,
                    log
            )
        }
    }

    override fun isValidForLookup(): Boolean = true
}
