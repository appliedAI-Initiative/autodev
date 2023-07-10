# AutoDev IntellIJ IDEA Plugin

A plugin for IntelliJ IDEA and other JetBrains IDEs.

## Development Environment

When opening this folder as a project in IntelliJ IDEA; 
a gradle project should be detected and the run configuration `Run Plugin`
should be available.
Running this configuration will start an IntelliJ instance with the plugin enabled.

### Building Dependencies

The project depends on components from TabNine.
To build these dependencies and make them available on your local machine, 
checkout branch `v1.0.9-autodev` from
[here](https://github.com/opcode81/tabnine-intellij/tree/v1.0.9-autodev).
Then, in folders `Common` and `Tabnine`, run

     ../gradlew -x test build publishToMavenLocal

NOTE: Use a newer version of JDK 11 (e.g. 11.0.9), otherwise it may not work
(it didn't work for me with either JDK 17 or JDK 11.0.0).

## Components

* The resource file `plugin.xml` defines the plugin components that are activated.
* The package `de.appliedai.autodev.autocomplete` handles the auto-completion.
* The package `de.appliedai.autodev.actions` contains editor actions (available in the editor context menu).
* Class `de.appliedai.autodev.ServiceClient` contains the service client implementation.
* Class `de.appliedai.autodev.AutoDevToolWindowManager` manages the creation of tool window components/tabs that are displayed in reaction to user queries.

