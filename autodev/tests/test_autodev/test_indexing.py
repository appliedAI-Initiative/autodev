from datetime import datetime
from typing import List

from autodev.indexing import FileInfo, scan_directory


class TestIndexing:
    def get_file_info_paths(self, file_infos: List[FileInfo]) -> List[str]:
        return [file_info.path for file_info in file_infos]

    def test_empty_directory(self, tmp_path):
        result = list(scan_directory(tmp_path))
        assert len(result) == 0

    def test_skip_hidden_dirs(self, tmp_path):
        (tmp_path / "test1.txt").write_text("test file 1\nsecond line")
        (tmp_path / "test2.md").write_text("# Title\nSome content\n")
        (tmp_path / ".hidden_dir").mkdir()
        (tmp_path / "ignored_dir").mkdir()
        (tmp_path / "ignored_dir" / "ignored.txt").write_text("ignored file")

        result = list(
            scan_directory(
                tmp_path, skip_hidden_dirs=True, ignored_dir_regex=r"ignored.*"
            )
        )
        assert len(result) == 2
        paths = self.get_file_info_paths(result)
        assert str(tmp_path / "test1.txt") in paths
        assert str(tmp_path / "test2.md") in paths

    def test_include_hidden_dirs(self, tmp_path):
        (tmp_path / "test1.txt").write_text("test file 1\nsecond line")
        (tmp_path / "test2.md").write_text("# Title\nSome content\n")
        (tmp_path / ".hidden_dir").mkdir()
        (tmp_path / ".hidden_dir" / "not-ignored.txt").write_text(
            "no longer ignored file"
        )

        result = list(scan_directory(tmp_path, skip_hidden_dirs=False))
        assert len(result) == 3
        paths = self.get_file_info_paths(result)
        assert str(tmp_path / "test1.txt") in paths
        assert str(tmp_path / "test2.md") in paths
        assert str(tmp_path / ".hidden_dir" / "not-ignored.txt") in paths

    def test_ignore_dir_regex(self, tmp_path):
        (tmp_path / "test1.txt").write_text("test file 1\nsecond line")
        (tmp_path / "test2.md").write_text("# Title\nSome content\n")
        (tmp_path / "ignored_dir").mkdir()
        (tmp_path / "ignored_dir" / "ignored.txt").write_text("ignored file")

        result = list(scan_directory(tmp_path, ignored_dir_regex=r"ignored.*"))
        assert len(result) == 2
        paths = self.get_file_info_paths(result)
        assert str(tmp_path / "test1.txt") in paths
        assert str(tmp_path / "test2.md") in paths

    def test_ignore_file_regex(self, tmp_path):
        (tmp_path / "test1.txt").write_text("test file 1\nsecond line")
        (tmp_path / "test2.md").write_text("# Title\nSome content\n")

        result = list(scan_directory(tmp_path, ignored_file_regex=r".*\.md"))
        assert len(result) == 1
        paths = self.get_file_info_paths(result)
        assert str(tmp_path / "test1.txt") in paths

    def test_last_modified_after(self, tmp_path_factory):
        before_creation_time = datetime.now()
        test_path = tmp_path_factory.mktemp("test_last_modified_after")
        (test_path / "test1.txt").write_text("test file 1\nsecond line")
        (test_path / "test2.md").write_text("# Title\nSome content\n")

        result = list(
            scan_directory(test_path, last_modified_after=before_creation_time)
        )
        assert len(result) == 2
        paths = self.get_file_info_paths(result)
        assert str(test_path / "test1.txt") in paths

        result = list(scan_directory(test_path, last_modified_after=datetime.now()))
        assert len(result) == 0

    def test_file_info_attributes(self, tmp_path):
        (tmp_path / "test1.txt").write_text("test file 1\nsecond line")
        (tmp_path / "test2.md").write_text("# Title\nSome content\n")

        result = list(scan_directory(tmp_path))
        assert len(result) == 2

        file1, file2 = result

        assert file1.extension == ".txt"
        assert file1.line_count == 2
        assert file1.last_modified is not None

        assert file2.extension == ".md"
        assert file2.line_count == 2
        assert file2.last_modified is not None
