package com.tabnineCommon.prediction;

import com.intellij.codeInsight.completion.CompletionParameters;
import com.intellij.openapi.application.ex.ApplicationUtil;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.fileEditor.FileDocumentManager;
import com.intellij.openapi.progress.ProgressManager;
import com.intellij.openapi.util.TextRange;
import com.intellij.openapi.vfs.VirtualFile;
import com.intellij.util.ObjectUtils;
import com.tabnineCommon.binary.BinaryRequestFacade;
import com.tabnineCommon.binary.exceptions.BinaryCannotRecoverException;
import com.tabnineCommon.binary.requests.autocomplete.AutocompleteRequest;
import com.tabnineCommon.binary.requests.autocomplete.AutocompleteResponse;
import com.tabnineCommon.binary.requests.autocomplete.ResultEntry;
import com.tabnineCommon.capabilities.SuggestionsModeService;
import com.tabnineCommon.inline.CompletionAdjustment;
import de.appliedai.autodev.AutoDevConfig;
import de.appliedai.autodev.ServiceClient;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import static com.tabnineCommon.general.StaticConfig.*;

public class CompletionFacade {
  private final SuggestionsModeService suggestionsModeService;
  private final ServiceClient serviceClient;

  public CompletionFacade(BinaryRequestFacade b, SuggestionsModeService suggestionsModeService) {
    this.suggestionsModeService = suggestionsModeService;
    this.serviceClient = new ServiceClient();
  }

  @Nullable
  public AutocompleteResponse retrieveCompletions(
      CompletionParameters parameters, @Nullable Integer tabSize) {
    try {
      String filename = getFilename(parameters.getOriginalFile().getVirtualFile());
      return ApplicationUtil.runWithCheckCanceled(
          () ->
              retrieveCompletions(
                  parameters.getEditor(), parameters.getOffset(), filename, tabSize, null),
          ProgressManager.getInstance().getProgressIndicator());
    } catch (BinaryCannotRecoverException e) {
      throw e;
    } catch (Exception e) {
      return null;
    }
  }

  @Nullable
  public AutocompleteResponse retrieveCompletions(
      @NotNull Editor editor,
      int offset,
      @Nullable Integer tabSize,
      @Nullable CompletionAdjustment completionAdjustment) {
    try {
      String filename =
          getFilename(FileDocumentManager.getInstance().getFile(editor.getDocument()));
      return retrieveCompletions(editor, offset, filename, tabSize, completionAdjustment);
    } catch (BinaryCannotRecoverException e) {
      throw e;
    } catch (Exception e) {
      return null;
    }
  }

  @Nullable
  public static String getFilename(@Nullable VirtualFile file) {
    return ObjectUtils.doIfNotNull(file, VirtualFile::getPath);
  }

  private static AutocompleteResponse createAutocompleteResponse(String completionPrefix, String completionSuffix) {
    AutocompleteResponse response = new AutocompleteResponse();
    response.old_prefix = "";
    var entry = new ResultEntry();
    entry.new_prefix = completionPrefix; //resultSet.getPrefixMatcher().getPrefix();
    entry.old_suffix = "";
    entry.new_suffix = completionSuffix;
    response.results = new ResultEntry[]{entry};
    return response;
  }

  @Nullable
  private AutocompleteResponse retrieveCompletions(
      @NotNull Editor editor,
      int offset,
      @Nullable String filename,
      @Nullable Integer tabSize,
      @Nullable CompletionAdjustment completionAdjustment) {
    Document document = editor.getDocument();

    int begin = Integer.max(0, offset - MAX_OFFSET);
    int end = Integer.min(document.getTextLength(), offset + MAX_OFFSET);
    AutocompleteRequest req = new AutocompleteRequest();
    req.before = document.getText(new TextRange(begin, offset));
    req.after = document.getText(new TextRange(offset, end));
    req.filename = filename;
    req.maxResults = MAX_COMPLETIONS;
    req.regionIncludesBeginning = (begin == 0);
    req.regionIncludesEnd = (end == document.getTextLength());
    req.offset = offset;
    req.line = document.getLineNumber(offset);
    req.character = offset - document.getLineStartOffset(req.line);
    req.indentation_size = tabSize;

    if (completionAdjustment != null) {
      completionAdjustment.adjustRequest(req);
    }

    AutocompleteResponse autocompleteResponse;
    if (AutoDevConfig.useDummyCompletions) {
      autocompleteResponse = createAutocompleteResponse("foobar(", ")");
    }
    else {
      String middle;
      try {
        middle = serviceClient.callAutoComplete(req.before, req.after, req.filename);
      } catch (Exception e) {
        System.err.println("Exception in service client: " + e);
        throw new RuntimeException(e);
      }
      autocompleteResponse = createAutocompleteResponse(middle, "");
    }

    if (completionAdjustment != null) {
      completionAdjustment.adjustResponse(autocompleteResponse);
    }

    return autocompleteResponse;
  }

  private int determineTimeoutBy(@NotNull String before) {
    if (!suggestionsModeService.getSuggestionMode().isInlineEnabled()) {
      return COMPLETION_TIME_THRESHOLD;
    }

    int lastNewline = before.lastIndexOf("\n");
    String lastLine = lastNewline >= 0 ? before.substring(lastNewline) : "";
    boolean endsWithWhitespacesOnly = lastLine.trim().isEmpty();
    return endsWithWhitespacesOnly ? NEWLINE_COMPLETION_TIME_THRESHOLD : COMPLETION_TIME_THRESHOLD;
  }
}
