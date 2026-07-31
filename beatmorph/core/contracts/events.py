"""核心数据契约 — 跨模块共享的数据类型。

这是整个系统的「通用语言」。所有模块（音频编码、Tokenizer、生成、
解码、后处理、导出）之间传递的数据都必须使用本文件定义的类型，
确保接口契约稳定、可静态校验、可序列化。

奠基文档约定：
- 每个 Note 的最小表示为 (time, lane, type, duration)
- Pattern Token 序列按「小节」粒度离散化
- 时间分辨率为 25Hz（MERT 帧率），谱面 Note 以毫秒或秒计

相关文档：docs/BasePlan.md §2、§3.7；docs/plans/00-core-contracts.md
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt


class NoteType(IntEnum):
    """Note 类型（4K VSRG 首批支持，预留扩展）。"""

    TAP = 0  # 普通键
    HOLD = 1  # 长条（含 duration）
    MINE = 2  # 地雷（避让）
    ROLL = 3  # 滚动长条（部分格式）
    FAKE = 4  # 假键


class GameMode(IntEnum):
    """目标游戏模式。Phase 1 仅 MANIA_4K。"""

    MANIA_4K = 4  # 4K VSRG（主攻）
    MANIA_5K = 5  # 5K VSRG（扩展，数据侧诚实标记，见 RFC-0025）
    MANIA_6K = 6  # 6K VSRG（扩展）
    MANIA_7K = 7  # 7K VSRG（扩展）
    MANIA_8K = 8  # 8K VSRG（扩展）
    OSU_STD = 0  # osu! standard（扩展）
    MAIMAI = 99  # maimai (.ma2，扩展)


class Note(BaseModel):
    """单条 Note 的规范表示。

    所有模块内部一律用该结构；写入特定格式时由 writer 转换。

    Attributes:
        time: 击打时刻，单位 秒（浮点，便于对齐音频）。
        lane: 键位索引，0-based（4K 即 0..3）。
        type: Note 类型，见 :class:`NoteType`。
        duration: 仅 HOLD/ROLL 有效，长度（秒）；TAP/MINE 固定为 0。
    """

    time: NonNegativeFloat
    lane: NonNegativeInt
    type: NoteType = NoteType.TAP
    duration: NonNegativeFloat = 0.0

    def is_hold(self) -> bool:
        return self.type in (NoteType.HOLD, NoteType.ROLL)


class Section(BaseModel):
    """一个结构段落（Section），Stage1 规划的最小单元。

    奠基文档：每 4 小节一个 Section，输出目标密度/能量/段落类型。
    """

    index: NonNegativeInt
    start_time: NonNegativeFloat  # 秒
    end_time: NonNegativeFloat  # 秒
    bar_count: NonNegativeInt = 4  # 小节数（默认 4）
    density_target: float = Field(ge=0.0, le=1.0, description="目标密度 0-1")
    energy_level: float = Field(ge=0.0, le=1.0, description="能量等级 0-1")
    rest_probability: float = Field(ge=0.0, le=1.0, description="休息段概率")
    sections_type: str = Field(
        default="verse", description="段落类型:intro/verse/chorus/bridge/outro"
    )


class BpmPoint(BaseModel):
    """BPM 变速时间点（RFC-0005）。

    常速曲退化为单元素列表 ``[BpmPoint(time=0.0, bpm=...)]``。变速曲按
    「时间→BPM」的分段常数序列描述：某点生效后，至下一变速点前的整段均
    使用该点 BPM。时间单位为秒，与 :class:`Note` 对齐音频轴。

    .. note::
        小节边界由 ``bpm_points`` 与时间共同推得（奠基 §3.2.1「以 BPM 动态
        对齐」），而非预先存储。详见 Plan 02 栅格化逻辑。

    Attributes:
        time: 该 BPM 生效的起始时刻（秒，>=0）。
        bpm: 该段时间的 BPM（>0）。
    """

    time: NonNegativeFloat
    bpm: float = Field(gt=0.0, description="每分钟节拍数")


class Chart(BaseModel):
    """一张完整谱面的规范中间表示（IR，与格式无关）。

    上游：Stage3 解码器输出；下游：Stage4 后处理 + 格式 Writer。

    BPM 以变速时间点序列 ``bpm_points`` 表示（RFC-0005）。常速曲仅需
    ``[BpmPoint(time=0.0, bpm=...)]`` 单元素列表；变速曲按 time 升序追加。
    需要标量主 BPM 的场景（如 RAG 元数据过滤）调用 :meth:`primary_bpm`。

    Attributes:
        version: IR 版本，用于不兼容变更兼容性判断。
        mode: 游戏模式（决定键位数）。
        difficulty: 难度等级 1-15（奠基文档约束）。
        bpm_points: BPM 变速时间点序列（至少 1 个，按 time 升序）。
        audio_path: 关联音频文件（仅运行期使用，不序列化敏感音频）。
        title: 曲目标题。
        artist: 艺术家。
        notes: 全部 Note 列表，按 time 升序。
        sections: 全局规划生成的段落信息（可选）。
        meta: 自由扩展元数据。
    """

    version: str = "ir-1"
    mode: GameMode = GameMode.MANIA_4K
    difficulty: int = Field(ge=1, le=15)
    bpm_points: list[BpmPoint] = Field(min_length=1)
    audio_path: str | None = None
    title: str = "untitled"
    artist: str = "unknown"
    notes: list[Note] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    meta: dict[str, str | int | float] = Field(default_factory=dict)

    def primary_bpm(self) -> float:
        """返回标量主 BPM。

        取 ``bpm_points`` 首个点的 BPM。常速曲即唯一 BPM；变速曲此前可由
        解析器排序使主段置于首位。下游需要单值 BPM 的场景（RAG 过滤、
        元数据展示等）调用本方法，无需手写遍历。
        """
        return self.bpm_points[0].bpm

    def lane_count(self) -> int:
        """当前模式的键位数。"""
        if self.mode == GameMode.OSU_STD:
            return 0  # osu!std 非键位模型
        return int(self.mode)

    def sorted_notes(self) -> list[Note]:
        """返回按时间升序的 Note（保证确定性）。"""
        return sorted(self.notes, key=lambda n: (n.time, n.lane))


class PatternToken(BaseModel):
    """VQ-VAE 编码后的离散 Pattern Token。

    一个 Token 对应一个小节内的 Note 集合模式（奠基文档 §3.2）。
    """

    code: NonNegativeInt  # VQ 码本索引
    bar_index: NonNegativeInt  # 所属小节序号
    start_time: NonNegativeFloat  # 该小节起点（秒）
    duration_bars: NonNegativeInt = 1  # 覆盖小节数（默认 1）
