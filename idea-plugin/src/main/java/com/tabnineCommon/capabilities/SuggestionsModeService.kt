package com.tabnineCommon.capabilities

import de.appliedai.autodev.AutoDevConfig

class SuggestionsModeService {
    fun getSuggestionMode(): SuggestionsMode {
        return AutoDevConfig.suggestionsMode;
    }
}
