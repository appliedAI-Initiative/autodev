package de.appliedai.autodev;

import com.intellij.util.concurrency.AppExecutorUtil;

import java.util.concurrent.TimeUnit;

/**
 * This component  ensures that within a given time period of windowMs, at most one task is executed, the
 * latest one.
 * It does so by delaying the execution of the tasks it is given until at least a windowMs milliseconds have
 * passed since the last task was submitted (if any). Then it checks whether the given task is still the latest
 * task and discards it if it is not.
 */
public class LatestTaskInWindowExecutor {
    private final int windowMs;
    private Long lastRequestTimeMs = null;
    private long nextTaskId = 1;

    public LatestTaskInWindowExecutor(int windowMs) {
        this.windowMs = windowMs;
    }

    public void submitTask(Runnable fn, TaskLogger log) {
        // determine time to wait until we can decide whether to run this task
        long requiredDelayMs;
        long requestTimeMs = System.currentTimeMillis();
        if (lastRequestTimeMs != null) {
            long timePassedMs = requestTimeMs - lastRequestTimeMs;
            if (timePassedMs > this.windowMs)
                requiredDelayMs = 0;
            else
                requiredDelayMs = windowMs - timePassedMs;
        }
        else {
            requiredDelayMs = windowMs;
        }

        // update last execution time
        lastRequestTimeMs = requestTimeMs;

        // assign task id
        long taskId = nextTaskId++;

        Runnable checkTaskExecution = () -> {
            long activeTaskId = nextTaskId - 1;
            if (activeTaskId != taskId) {
                log.info(String.format("Task has evaporated; activeTaskId=%d, taskId=%d", activeTaskId, taskId));
            }
            else {
                log.info("Running task");
                fn.run();
            }
        };

        runDelayed(requiredDelayMs, checkTaskExecution, log);
    }

    private void runDelayed(long delayMs, Runnable runnable, TaskLogger log) {
        if (delayMs == 0)
            runnable.run();
        else {
            log.info("Delaying task execution check by " + delayMs + " ms");
            AppExecutorUtil.getAppScheduledExecutorService().schedule(runnable, delayMs, TimeUnit.MILLISECONDS);
        }
    }
}
