# Changelog

Perubahan penting pada service Astra RAG didokumentasikan di file ini.

## [Unreleased]

### Added

- **Trusted Routing:** Menambahkan schema routing yang ketat untuk provider, model ID, model version, policy version, token limits, dan temperature.
- **Generation Context:** Menambahkan generation ID, user ID, dan routing snapshot pada request/query state.
- **Usage Events:** Menambahkan event usage provider untuk embedding, grader, dan answer call.
- **Usage SSE:** Menambahkan streaming event usage yang terurut serta aggregate usage pada status completed, cancelled, dan error.
- **Provider Cache:** Menambahkan cache client ChatOpenAI yang dibatasi berdasarkan provider, model, dan version.
- **Tests:** Menambahkan test untuk query stream, routing validation, node usage events, dan aggregate usage.

### Changed

- **Query Contract:** Request query kini menerima generation ID, user ID, dan trusted routing snapshot dari backend.
- **RAG Routing:** Embedding, grader, dan answer call menggunakan routing yang disediakan backend, bukan model arbitrary dari user request.
- **Fallback Behavior:** Memperbaiki broaden/fallback flow agar usage event dan terminal stream state tetap konsisten.
- **Ingestion Billing:** Transcript dan PDF ingestion ditandai sebagai platform spend (`billable_to_user=false`), bukan usage yang dibebankan ke user generation.
- **Schema Strictness:** Memperketat schema request dan response untuk menjaga konsistensi lintas backend, Astra, dan frontend.

### Fixed

- **Usage Ordering:** Memastikan usage event dikirim sebelum aggregate terminal event pada stream.
- **Route Validation:** Menolak routing yang tidak lengkap atau tidak sesuai provider/model version yang dipercaya.
- **Stream Terminal State:** Menyamakan handling completed, cancelled, dan error agar settlement backend menerima status yang deterministik.

### Verification

- 9 Astra pytest tests passed.
- Ruff check passed.
- Python compileall passed.
