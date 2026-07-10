"""Tests for the level-0 file hash."""

from audiotwin import file_hash


def test_identical_files_hash_equal(sine_440, sine_440_copy):
    assert file_hash(sine_440) == file_hash(sine_440_copy)


def test_different_files_hash_differ(sine_440, different_audio):
    assert file_hash(sine_440) != file_hash(different_audio)


def test_hash_is_stable(sine_440):
    assert file_hash(sine_440) == file_hash(sine_440)


def test_hash_is_sha256_hex(sine_440):
    digest = file_hash(sine_440)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
