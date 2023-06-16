package de.appliedai.autodev;


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

    public static TempLogger getInstance(Class<?> cls) {
        return new TempLogger(cls);
    }
}
