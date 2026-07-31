"""manifest 模块单元测试（离线，纯 stdlib）。

覆盖：JSONL 解析、坏行跳过、注入器命中/缺失/no-op/不覆盖/未 Ranked 注零。
"""

from __future__ import annotations

import json
from pathlib import Path

from beatmorph.data.manifest import (
    SetRecord,
    build_injector,
    load_manifest,
)

# ── 测试夹具构造 ──────────────────────────────────────────────


def _manifest_line(**overrides: object) -> str:
    """构造一行合法 manifest JSON。"""
    obj: dict[str, object] = {
        "sid": 1000,
        "title": "Test Song",
        "artist": "Test Artist",
        "creator": "TestMapper",
        "stars": 5.5,
        "play_count": 1500,
        "approved": 1,
        "modes": 8,
        "downloaded_at": "2026-07-31T00:00:00",
        "n_diffs": 4,
        "path": "1000",
        "unranked": False,
    }
    obj.update(overrides)
    return json.dumps(obj, ensure_ascii=False)


def _write_manifest(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestLoadManifest:
    def test_parses_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        _write_manifest(
            path,
            [
                _manifest_line(sid=1, stars=4.0),
                _manifest_line(sid=2, stars=6.5, play_count=9999),
                _manifest_line(sid=3, stars=3.2),
            ],
        )

        records = load_manifest(path)

        assert len(records) == 3
        assert set(records) == {1, 2, 3}
        assert isinstance(records[1], SetRecord)
        assert records[2].play_count == 9999
        assert records[2].stars == 6.5

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        _write_manifest(
            path,
            [
                _manifest_line(sid=1),
                "this is not json {{{",
                "",  # 空行应被忽略
                _manifest_line(sid=2, stars=7.0),
                json.dumps({"no_sid": "missing required"}),  # 缺 sid
            ],
        )

        records = load_manifest(path)

        assert set(records) == {1, 2}
        assert records[2].stars == 7.0

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        records = load_manifest(tmp_path / "nope.jsonl")
        assert records == {}

    def test_unranked_derived_from_zero_stars(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        _write_manifest(path, [_manifest_line(sid=1, stars=0.0, approved=3, unranked=True)])

        records = load_manifest(path)
        assert records[1].unranked is True
        assert records[1].stars == 0.0


class TestManifestInjector:
    def test_applies_when_sid_matches(self) -> None:
        records = {1000: _rec(sid=1000, stars=5.5, play_count=1500, approved=1)}
        inj = build_injector(records)

        meta: dict[str, str | int | float] = {"beatmap_set_id": 1000, "creator": "X"}
        applied = inj.apply(meta)

        assert applied is True
        assert meta["difficulty_rating"] == 5.5
        assert meta["playcount"] == 1500
        assert meta["license"] == "academic"
        assert meta["approved"] == 1
        # 既有键保留
        assert meta["creator"] == "X"

    def test_noop_when_sid_abscent_from_meta(self) -> None:
        inj = build_injector({1000: _rec(sid=1000)})
        meta: dict[str, str | int | float] = {"creator": "X"}  # 无 beatmap_set_id
        assert inj.apply(meta) is False
        assert meta == {"creator": "X"}  # 不变

    def test_noop_when_sid_not_in_manifest(self) -> None:
        inj = build_injector({1000: _rec(sid=1000)})
        meta: dict[str, str | int | float] = {"beatmap_set_id": 9999}
        assert inj.apply(meta) is False
        assert "difficulty_rating" not in meta

    def test_does_not_overwrite_existing(self) -> None:
        """调用方已设的值优先（只加不覆盖）。"""
        records = {1000: _rec(sid=1000, stars=5.5, play_count=1500)}
        inj = build_injector(records)

        meta: dict[str, str | int | float] = {
            "beatmap_set_id": 1000,
            "difficulty_rating": 9.0,  # 已有精确星级
            "playcount": 42,
            "license": "commercial",
        }
        inj.apply(meta)

        assert meta["difficulty_rating"] == 9.0  # 未被覆盖
        assert meta["playcount"] == 42
        assert meta["license"] == "commercial"
        # approved 仍注入（之前缺失）
        assert meta["approved"] == 1

    def test_unranked_injects_zero_stars(self) -> None:
        """未 Ranked (order=0.0) → 注入 0.0，下游 0.0<3.0 剔除。"""
        records = {1000: _rec(sid=1000, stars=0.0, approved=3, unranked=True)}
        inj = build_injector(records)

        meta: dict[str, str | int | float] = {"beatmap_set_id": 1000}
        inj.apply(meta)

        assert meta["difficulty_rating"] == 0.0
        # 模拟 _quality_filter 的拒判逻辑
        assert float(meta["difficulty_rating"]) < 3.0

    def test_non_int_sid_noop(self) -> None:
        """beatmap_set_id 非 int（如 None/str）→ no-op。"""
        inj = build_injector({1000: _rec(sid=1000)})
        meta: dict[str, str | int | float] = {"beatmap_set_id": "1000"}
        assert inj.apply(meta) is False

    def test_len(self) -> None:
        inj = build_injector({1: _rec(sid=1), 2: _rec(sid=2)})
        assert len(inj) == 2


# ── 辅助 ──────────────────────────────────────────────────────


def _rec(
    sid: int,
    stars: float = 5.0,
    play_count: int = 1000,
    approved: int = 1,
    unranked: bool = False,
) -> SetRecord:
    return SetRecord(
        sid=sid,
        title="t",
        artist="a",
        creator="c",
        stars=stars,
        play_count=play_count,
        approved=approved,
        modes=8,
        downloaded_at="2026-07-31T00:00:00",
        n_diffs=4,
        path=str(sid),
        unranked=unranked,
    )
