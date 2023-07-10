package de.appliedai.autodev.autocomplete;

import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.diagnostic.Logger;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.fileEditor.FileDocumentManager;
import com.intellij.util.ObjectUtils;
import com.tabnineCommon.binary.BinaryRequestFacade;
import com.tabnineCommon.binary.requests.autocomplete.AutocompleteResponse;
import com.tabnineCommon.binary.requests.notifications.shown.SnippetShownRequest;
import com.tabnineCommon.binary.requests.notifications.shown.SuggestionDroppedReason;
import com.tabnineCommon.binary.requests.notifications.shown.SuggestionShownRequest;
import com.tabnineCommon.capabilities.SuggestionsMode;
import com.tabnineCommon.capabilities.SuggestionsModeService;
import com.tabnineCommon.general.CompletionKind;
import com.tabnineCommon.general.CompletionsEventSender;
import com.tabnineCommon.general.DependencyContainer;
import com.tabnineCommon.general.SuggestionTrigger;
import com.tabnineCommon.inline.*;
import com.tabnineCommon.inline.render.GraphicsUtilsKt;
import com.tabnineCommon.intellij.completions.CompletionUtils;
import com.tabnineCommon.prediction.CompletionFacade;
import com.tabnineCommon.prediction.TabNineCompletion;
import de.appliedai.autodev.AutoDevConfig;
import de.appliedai.autodev.util.LatestTaskInWindowExecutor;
import de.appliedai.autodev.util.TaskLogger;
import de.appliedai.autodev.util.TempLogger;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.tabnineCommon.general.Utils.executeThread;
import static com.tabnineCommon.general.Utils.isUnitTestMode;
import static com.tabnineCommon.prediction.CompletionFacade.getFilename;

public class InlineCompletionHandler {
  private final CompletionFacade completionFacade;
  private final BinaryRequestFacade binaryRequestFacade;
  private final SuggestionsModeService suggestionsModeService;
  private final CompletionsEventSender completionsEventSender =
      DependencyContainer.instanceOfCompletionsEventSender();
  private Future<?> lastDebounceRenderTask = null;
  private Future<?> lastFetchAndRenderTask = null;
  private Future<?> lastFetchInBackgroundTask = null;
  private long taskId = 1;
  private LatestTaskInWindowExecutor documentUpdateTaskExecutor = new LatestTaskInWindowExecutor(500);

  private final TempLogger log = TempLogger.getInstance(InlineCompletionHandler.class);
  private static InlineCompletionHandler instance = null;

  public InlineCompletionHandler(
      CompletionFacade completionFacade,
      BinaryRequestFacade binaryRequestFacade,
      SuggestionsModeService suggestionsModeService) {
    this.completionFacade = completionFacade;
    this.binaryRequestFacade = binaryRequestFacade;
    this.suggestionsModeService = suggestionsModeService;
  }

  public static synchronized InlineCompletionHandler getInstance() {
    if (instance == null) {
      instance = new InlineCompletionHandler(DependencyContainer.instanceOfCompletionFacade(),
              DependencyContainer.instanceOfBinaryRequestFacade(),
              DependencyContainer.instanceOfSuggestionsModeService());
    }
    return instance;
  }

  public void retrieveAndShowCompletion(
      @NotNull Editor editor,
      int offset,
      @Nullable TabNineCompletion lastShownSuggestion,
      @NotNull String userInput,
      @NotNull CompletionAdjustment completionAdjustment,
      boolean isManualRequest,
      TaskLogger log) {
    Integer tabSize = GraphicsUtilsKt.getTabSize(editor);

    log.info("Cancelling tasks");
    ObjectUtils.doIfNotNull(lastFetchInBackgroundTask, task -> task.cancel(false));
    ObjectUtils.doIfNotNull(lastFetchAndRenderTask, task -> task.cancel(false));
    ObjectUtils.doIfNotNull(lastDebounceRenderTask, task -> task.cancel(false));

    // NOTE: Retrieving adjusted completions doesn't work for manual requests as the userInput is "" and the prefix
    // of the stored completion is typically also ""
    // TODO The root of the problem is probably that the adjusted completions aren't stored (only the original ones are)
    if (!isManualRequest) {
      log.info("Retrieve adjusted completions");
      List<TabNineCompletion> cachedCompletions =
              InlineCompletionCache.getInstance().retrieveAdjustedCompletions(editor, userInput);
      if (!cachedCompletions.isEmpty()) {
        var firstCompletion = cachedCompletions.get(0);
        log.info(String.format("Showing cached completions: userInput='%s', suffix='%s'", userInput, firstCompletion.getSuffix()));
        renderCachedCompletions(editor, offset, tabSize, cachedCompletions, completionAdjustment, log);
        return;
      }
    }

    if (lastShownSuggestion != null) {
      log.info("send suggestion dropped");
      SuggestionDroppedReason reason =
          completionAdjustment instanceof LookAheadCompletionAdjustment
              ? SuggestionDroppedReason.ScrollLookAhead
              : SuggestionDroppedReason.UserNotTypedAsSuggested;
      log.info("Last shown suggestion dropped: reason=" + reason);
      // if the last rendered suggestion is not null, this means that the user has typed something
      // that doesn't match the previous suggestion - hence the reason is `UserNotTypedAsSuggested`
      // (or `ScrollLookAhead` if the suggestion's source is from look-ahead).
      completionsEventSender.sendSuggestionDropped(editor, lastShownSuggestion, reason);
    }

    boolean retrieveNewCompletions = AutoDevConfig.autoRequestCompletionsOnDocumentChange || isManualRequest;
    if (retrieveNewCompletions) {
      Runnable renderNewCompletionsTask = () -> {
        ApplicationManager.getApplication()
                .invokeLater(
                        () -> {
                          log.info("renderNewCompletions");
                          renderNewCompletions(
                                  editor,
                                  tabSize,
                                  getCurrentEditorOffset(editor, userInput),
                                  editor.getDocument().getModificationStamp(),
                                  completionAdjustment,
                                  log);
                        });
      };

      if (isManualRequest)
        renderNewCompletionsTask.run();
      else {
        // For changes due to document edits, make sure that only one (the latest) task in any given time window
        // will be run to avoid excessive queries
        documentUpdateTaskExecutor.submitTask(renderNewCompletionsTask, log);
      }
    }
    else {
      log.info("Not retrieving new completions because auto-retrieval on document change is disabled");
    }
  }

  private void renderCachedCompletions(
          @NotNull Editor editor,
          int offset,
          Integer tabSize,
          @NotNull List<TabNineCompletion> cachedCompletions,
          @NotNull CompletionAdjustment completionAdjustment, TaskLogger log) {
    showInlineCompletion(editor, cachedCompletions, offset, null);
    // NOTE: If we can show cached completions, it seems unnecessary to fetch new completions in the background,
    // so this is disabled
    /*
    log.info("Assigning lastFetchInBackgroundTask");
    lastFetchInBackgroundTask =
        executeThread(
            () -> retrieveInlineCompletion(editor, offset, tabSize, completionAdjustment, log));
    */
  }

  private int getCurrentEditorOffset(@NotNull Editor editor, @NotNull String userInput) {
    return editor.getCaretModel().getOffset()
        + (ApplicationManager.getApplication().isUnitTestMode() ? userInput.length() : 0);
  }

  private void renderNewCompletions(
          @NotNull Editor editor,
          Integer tabSize,
          int offset,
          long modificationStamp,
          @NotNull CompletionAdjustment completionAdjustment,
          TaskLogger log) {
    log.info("Assigning lastFetchAndRenderTask");
    lastFetchAndRenderTask =
        executeThread(
            () -> {
              log.info("update last completion request");
              CompletionTracker.updateLastCompletionRequestTime(editor);
              log.info("retrieve inline completion");
              List<TabNineCompletion> beforeDebounceCompletions =
                  retrieveInlineCompletion(editor, offset, tabSize, completionAdjustment, log);
              long debounceTime = CompletionTracker.calcDebounceTime(editor, completionAdjustment);
              log.info("Debounce time: " + debounceTime);
              if (debounceTime == 0) {
                log.info("No debounce: rerenderCompletion");
                rerenderCompletion(
                    editor,
                    beforeDebounceCompletions,
                    offset,
                    modificationStamp,
                    completionAdjustment, log);
              }
              else {
                log.info("Assigning lastDebounceRenderTask");
                lastDebounceRenderTask =
                        executeThread(
                                () -> {
                                  log.info("After debounce");
                                  List<TabNineCompletion> completions =
                                          retrieveInlineCompletion(editor, offset, tabSize, completionAdjustment, log);
                                  rerenderCompletion(
                                          editor, completions, offset, modificationStamp, completionAdjustment, log);
                                },
                                debounceTime,
                                TimeUnit.MILLISECONDS);
              }
            });
  }

  private void rerenderCompletion(
          @NotNull Editor editor,
          List<TabNineCompletion> completions,
          int offset,
          long modificationStamp,
          @NotNull CompletionAdjustment completionAdjustment,
          TaskLogger log) {
    log.info("rerenderCompletion");
    ApplicationManager.getApplication()
        .invokeLater(
            () -> {
              if (shouldCancelRendering(editor, modificationStamp, offset)) {
                log.info("Rendering cancelled");
                return;
              }
              if (shouldRemovePopupCompletions(completionAdjustment)) {
                completions.removeIf(completion -> !completion.isSnippet());
              }
              log.info("Showing inline completions");
              showInlineCompletion(
                  editor,
                  completions,
                  offset,
                  (completion) -> afterCompletionShown(completion, editor));
            });
  }

  private boolean shouldCancelRendering(
      @NotNull Editor editor, long modificationStamp, int offset) {
    if (isUnitTestMode()) {
      return false;
    }
    boolean isModificationStampChanged =
        modificationStamp != editor.getDocument().getModificationStamp();
    boolean isOffsetChanged = offset != editor.getCaretModel().getOffset();
    return isModificationStampChanged || isOffsetChanged;
  }

  /**
   * remove popup completions when 1. the suggestion mode is HYBRID and 2. the completion adjustment
   * type is not LookAhead
   */
  private boolean shouldRemovePopupCompletions(@NotNull CompletionAdjustment completionAdjustment) {
    return suggestionsModeService.getSuggestionMode() == SuggestionsMode.HYBRID
        && completionAdjustment.getSuggestionTrigger() != SuggestionTrigger.LookAhead;
  }

  private List<TabNineCompletion> retrieveInlineCompletion(
      @NotNull Editor editor,
      int offset,
      Integer tabSize,
      @NotNull CompletionAdjustment completionAdjustment,
      TaskLogger log) {
    log.info("completionFacade.retrieveCompletions");
    AutocompleteResponse completionsResponse =
        this.completionFacade.retrieveCompletions(editor, offset, tabSize, completionAdjustment);

    if (completionsResponse == null || completionsResponse.results.length == 0) {
      log.info("Completions response is empty");
      return Collections.emptyList();
    }
    else {
      var list = createCompletions(
              completionsResponse,
              editor.getDocument(),
              offset,
              completionAdjustment.getSuggestionTrigger());
      log.info("Completions received");
      return list;
    }
  }

  private void showInlineCompletion(
      @NotNull Editor editor,
      List<TabNineCompletion> completions,
      int offset,
      @Nullable OnCompletionPreviewUpdatedCallback onCompletionPreviewUpdatedCallback) {
    if (completions.isEmpty()) {
      return;
    }
    InlineCompletionCache.getInstance().store(editor, completions);

    TabNineCompletion displayedCompletion =
        CompletionPreview.createInstance(editor, completions, offset);

    if (displayedCompletion == null) {
      return;
    }

    log.info("Showing completion: " + displayedCompletion.getSuffix());
    if (onCompletionPreviewUpdatedCallback != null) {
      onCompletionPreviewUpdatedCallback.onCompletionPreviewUpdated(displayedCompletion);
    }
  }

  private void afterCompletionShown(TabNineCompletion completion, Editor editor) {
    if (completion.completionMetadata == null) return;
    Boolean isCached = completion.completionMetadata.is_cached();
    // binary is not supporting api version ^4.0.57
    if (isCached == null) return;

    try {
      String filename =
          getFilename(FileDocumentManager.getInstance().getFile(editor.getDocument()));
      if (filename == null) {
        Logger.getInstance(getClass())
            .warn("Could not send SuggestionShown request. the filename is null");
        return;
      }
      this.binaryRequestFacade.executeRequest(
          new SuggestionShownRequest(
              completion.getNetLength(), filename, completion.completionMetadata));

      if (completion.completionMetadata.getCompletion_kind() == CompletionKind.Snippet
          && !isCached) {
        Map<String, Object> context = completion.completionMetadata.getSnippet_context();
        if (context == null) {
          Logger.getInstance(getClass())
              .warn("Could not send SnippetShown request. intent is null");
          return;
        }

        this.binaryRequestFacade.executeRequest(new SnippetShownRequest(filename, context));
      }
    } catch (RuntimeException e) {
      // swallow - nothing to do with this
    }
  }

  private List<TabNineCompletion> createCompletions(
      AutocompleteResponse completions,
      @NotNull Document document,
      int offset,
      SuggestionTrigger suggestionTrigger) {
    return IntStream.range(0, completions.results.length)
        .mapToObj(
            index ->
                CompletionUtils.createTabnineCompletion(
                    document,
                    offset,
                    completions.old_prefix,
                    completions.results[index],
                    index,
                    suggestionTrigger))
        .filter(completion -> completion != null && !completion.getSuffix().isEmpty())
        .collect(Collectors.toList());
  }
}
