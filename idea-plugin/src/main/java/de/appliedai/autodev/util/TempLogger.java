package de.appliedai.autodev.util;


import com.intellij.openapi.diagnostic.Logger;

public class TempLogger {
    private Logger log;

    public TempLogger(Class<?> cls) {
        log = Logger.getInstance(cls);
    }

    public void info(String message) {
        log.warn(message);
    }

    public void debug(String message) {
        log.warn(message);
    }

    public void error(String message) {
        log.error(message);
    }

    public static TempLogger getInstance(Class<?> cls) {
        return new TempLogger(cls);
    }
}
