// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

import android.util.Log
import java.io.File
import java.io.IOException
import java.nio.file.FileVisitResult
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.SimpleFileVisitor
import java.nio.file.StandardCopyOption
import java.nio.file.attribute.BasicFileAttributes

object DownloadFileUtils {
    const val TAG: String = "FileUtils"

    fun repoFolderName(repoId: String?, repoType: String?): String {
        if (repoId == null || repoType == null) {
            return ""
        }
        val repoParts = repoId.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
        val parts: MutableList<String> = ArrayList()
        parts.add(repoType + "s") // e.g., "models"
        for (part in repoParts) {
            if (part != null && !part.isEmpty()) {
                parts.add(part)
            }
        }
        // Join parts with "--" separator
        return java.lang.String.join("--", parts)
    }

    fun deleteDirectoryRecursively(dir: File?): Boolean {
        if (dir == null || !dir.exists()) {
            return false
        }

        val dirPath = dir.toPath()
        try {
            Files.walkFileTree(
                    dirPath,
                    object : SimpleFileVisitor<Path>() {
                        @Throws(IOException::class)
                        override fun visitFile(
                                file: Path,
                                attrs: BasicFileAttributes
                        ): FileVisitResult {
                            Files.delete(file)
                            return FileVisitResult.CONTINUE
                        }

                        @Throws(IOException::class)
                        override fun postVisitDirectory(
                                directory: Path,
                                exc: IOException?
                        ): FileVisitResult {
                            Files.delete(directory)
                            return FileVisitResult.CONTINUE
                        }

                        @Throws(IOException::class)
                        override fun visitFileFailed(
                                file: Path,
                                exc: IOException
                        ): FileVisitResult {
                            return FileVisitResult.TERMINATE
                        }
                    }
            )
            return true
        } catch (e: IOException) {
            return false
        }
    }

    fun copyDirectoryRecursively(sourceDir: Path, targetDir: Path) {
        Files.walkFileTree(
                sourceDir,
                object : SimpleFileVisitor<Path>() {
                    @Throws(IOException::class)
                    override fun preVisitDirectory(
                            dir: Path,
                            attrs: BasicFileAttributes
                    ): FileVisitResult {
                        val targetPath = targetDir.resolve(sourceDir.relativize(dir))
                        if (!Files.exists(targetPath)) {
                            Files.createDirectories(targetPath)
                        }
                        return FileVisitResult.CONTINUE
                    }

                    @Throws(IOException::class)
                    override fun visitFile(
                            file: Path,
                            attrs: BasicFileAttributes
                    ): FileVisitResult {
                        val targetPath = targetDir.resolve(sourceDir.relativize(file))
                        Files.copy(file, targetPath, StandardCopyOption.REPLACE_EXISTING)
                        return FileVisitResult.CONTINUE
                    }
                }
        )
    }

    fun getPointerPath(storageFolder: File?, commitHash: String, relativePath: String): File {
        val commitFolder = File(storageFolder, "snapshots/$commitHash")
        return File(commitFolder, relativePath)
    }

    @JvmStatic
    fun getPointerPathParent(storageFolder: File?, sha: String): File {
        return File(storageFolder, "snapshots/$sha")
    }

    fun getLastFileName(path: String): String {
        if (path.isEmpty()) {
            return path
        }
        val pos = path.lastIndexOf('/')
        return if ((pos == -1)) path else path.substring(pos + 1)
    }

    fun createSymlink(target: String?, linkPath: String?) {
        if (target == null || linkPath == null) return
        val targetPath = Paths.get(target)
        val link = Paths.get(linkPath)
        createSymlink(targetPath, link)
    }
    private val copyLock = Any()

    fun createSymlink(target: Path?, linkPath: Path?) {
        if (target == null || linkPath == null) return

        var symlinkCreated = false
        try {
            // First attempt: direct symlink
            Files.createSymbolicLink(linkPath, target)
            symlinkCreated = true
        } catch (e: java.nio.file.FileAlreadyExistsException) {
            Log.d(TAG, "Link path already exists: $linkPath. Attempting to resolve collision.")
            try {
                if (Files.isSymbolicLink(linkPath)) {
                    val existingTarget = Files.readSymbolicLink(linkPath)
                    if (existingTarget == target) {
                        symlinkCreated = true
                    } else {
                        Files.delete(linkPath)
                        Files.createSymbolicLink(linkPath, target)
                        symlinkCreated = true
                    }
                } else {
                    // It's a directory or a file, not a symlink
                    // We'll let the fallback handle it if it fails to delete/link
                    Files.delete(linkPath)
                    Files.createSymbolicLink(linkPath, target)
                    symlinkCreated = true
                }
            } catch (innerEx: Exception) {
                Log.w(TAG, "Collision resolution failed: ${innerEx.message}")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Direct symlink failed: ${e.message}")
        }

        if (!symlinkCreated) {
            // Fallback for filesystems that don't support symlinks (like Android external storage)
            Log.w(TAG, "Falling back to hard copy for $linkPath")
            synchronized(copyLock) {
                try {
                    if (Files.exists(linkPath)) {
                        if (Files.isDirectory(linkPath)) {
                            // Check if it's already a valid model dir to avoid redundant copies
                            // but usually since we are here, we want to ensure it's fresh
                            deleteDirectoryRecursively(linkPath.toFile())
                        } else {
                            Files.delete(linkPath)
                        }
                    }

                    if (Files.isDirectory(target)) {
                        Log.d(TAG, "Copying directory recursively from $target to $linkPath")
                        copyDirectoryRecursively(target, linkPath)
                    } else {
                        Files.copy(target, linkPath, StandardCopyOption.REPLACE_EXISTING)
                    }
                    symlinkCreated = true
                } catch (copyEx: Exception) {
                    Log.e(TAG, "Fallback copy failed for $linkPath", copyEx)
                }
            }
        }

        if (symlinkCreated) {
            try {
                val successFile =
                        if (Files.isDirectory(linkPath)) {
                            File(linkPath.toFile(), ".success")
                        } else {
                            File(linkPath.toFile().parentFile, "${linkPath.toFile().name}.success")
                        }
                if (!successFile.exists()) {
                    successFile.createNewFile()
                }
                Log.d(TAG, "Created success marker: ${successFile.absolutePath}")
            } catch (e: Exception) {
                Log.w(TAG, "Failed to create success marker: ${e.message}")
            }
        }
    }

    private fun handleSymlinkCollision(target: Path, linkPath: Path) {
        try {
            if (Files.isSymbolicLink(linkPath)) {
                val existingTarget = Files.readSymbolicLink(linkPath)
                if (existingTarget != target) {
                    Files.delete(linkPath)
                    Files.createSymbolicLink(linkPath, target)
                }
            } else {
                Files.delete(linkPath)
                Files.createSymbolicLink(linkPath, target)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to resolve symlink collision", e)
        }
    }
}
