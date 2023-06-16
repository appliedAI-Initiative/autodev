To build the Tabnine dependencies, use branch `v109` of the tabnine-intellij,
and in folders `Common` and `Tabnine` run

     ..\gradlew -x test build publishToMavenLocal

NOTE: Use a newer version of JDK 11 (e.g. 11.0.9), otherwise it may not work
(it didn't work for me with either JDK 17 or JDK 11.0.0).