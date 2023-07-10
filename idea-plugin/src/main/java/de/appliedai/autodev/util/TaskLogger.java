package de.appliedai.autodev.util;


public class TaskLogger {
    private String prefix;
    private TempLogger log;

    public TaskLogger(TempLogger log, String prefix) {
        this.prefix = prefix;
        this.log = log;
    }

    public void info(String message) {
        log.info(prefix + message);
    }

    public void error(String message) {
        log.error(prefix + message);
    }

    public void debug(String message) {
        log.debug(prefix + message);
    }
}
