plugins {
  id("java")
  id("org.jetbrains.kotlin.jvm") version "1.8.20"
  id("org.jetbrains.intellij") version "1.13.3"
}

group = "de.appliedai"
version = "1.0-SNAPSHOT"

repositories {
  mavenCentral()
  mavenLocal()
}

dependencies {
  implementation("org.apache.httpcomponents:httpcomponents-client:4.5.14")
  implementation("com.tabnineCommon:Common:1.0.9")
  implementation("com.tabnine:Tabnine:1.0.9")
}

// Configure Gradle IntelliJ Plugin
// Read more: https://plugins.jetbrains.com/docs/intellij/tools-gradle-intellij-plugin.html
intellij {
  version.set("2022.2.5")
  type.set("IC") // Target IDE Platform

  plugins.set(listOf(/* Plugin Dependencies */))
}

tasks {
  // Set the JVM compatibility versions
  withType<JavaCompile> {
    sourceCompatibility = "17"
    targetCompatibility = "17"
  }
  withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
    kotlinOptions.jvmTarget = "17"
  }

  patchPluginXml {
    sinceBuild.set("222")
    untilBuild.set("232.*")
  }

  signPlugin {
    certificateChain.set(System.getenv("CERTIFICATE_CHAIN"))
    privateKey.set(System.getenv("PRIVATE_KEY"))
    password.set(System.getenv("PRIVATE_KEY_PASSWORD"))
  }

  publishPlugin {
    token.set(System.getenv("PUBLISH_TOKEN"))
  }
}
