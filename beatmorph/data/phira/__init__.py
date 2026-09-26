"""Phira 数据获取层（plan 02 §3.2）：API 客户端 + 谱面包访问 + 清单 provenance。

- :mod:`beatmorph.data.phira.client` —— `GET /chart` 分页枚举、HTTP Range 预筛、
  单条目下载、`Provenance`/`Manifest` 落盘（M1 / M8）。
- :mod:`beatmorph.data.phira.package` —— `ChartPackage`：**只按 `info.yml.chart` 定位谱面文件**
  （M4，R1/R2/R4）。
"""

from beatmorph.data.phira.client import (
    CHART_FILE_FIELD,
    CHART_LIST_RESULTS_KEY,
    CHART_MUSIC_FIELD,
    CHART_PAGE_SIZE_MAX,
    CHART_TOTAL_EXPECTED,
    PHIRA_API_BASE,
    RANGE_PREFIX_BYTES,
    REQUEST_INTERVAL_S,
    ZIP_TAIL_BYTES,
    Manifest,
    ManifestError,
    ManifestPurpose,
    PhiraApiError,
    PhiraChartMeta,
    PhiraClient,
    PhiraZipError,
    Provenance,
    ZipEntry,
    ZipIndex,
    iter_meta_rows,
    provenance_for_api,
    read_manifest,
    sha1_file,
    sha1_hex,
    write_manifest,
)
from beatmorph.data.phira.package import (
    INFO_YML_NAME,
    ChartInfo,
    ChartPackage,
    ChartPackageError,
    chart_dest_path,
    normalize_chart_filename,
    parse_info_yaml,
)

__all__ = [
    "CHART_FILE_FIELD",
    "CHART_LIST_RESULTS_KEY",
    "CHART_MUSIC_FIELD",
    "CHART_PAGE_SIZE_MAX",
    "CHART_TOTAL_EXPECTED",
    "INFO_YML_NAME",
    "PHIRA_API_BASE",
    "RANGE_PREFIX_BYTES",
    "REQUEST_INTERVAL_S",
    "ZIP_TAIL_BYTES",
    "ChartInfo",
    "ChartPackage",
    "ChartPackageError",
    "Manifest",
    "ManifestError",
    "ManifestPurpose",
    "PhiraApiError",
    "PhiraChartMeta",
    "PhiraClient",
    "PhiraZipError",
    "Provenance",
    "ZipEntry",
    "ZipIndex",
    "chart_dest_path",
    "iter_meta_rows",
    "normalize_chart_filename",
    "parse_info_yaml",
    "provenance_for_api",
    "read_manifest",
    "sha1_file",
    "sha1_hex",
    "write_manifest",
]
