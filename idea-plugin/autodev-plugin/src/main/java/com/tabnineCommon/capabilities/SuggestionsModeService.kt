package com.tabnineCommon.capabilities

class SuggestionsModeService {
    fun getSuggestionMode(): SuggestionsMode {
        System.out.println("Called getSuggestionMode")
        return SuggestionsMode.HYBRID;
    }
}
