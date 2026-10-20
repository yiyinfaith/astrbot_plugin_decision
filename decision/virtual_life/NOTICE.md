# Source and modifications

Virtual Life core, templates, schema and engine are adapted from
https://github.com/FloranceYeh/astrbot_plugin_virtual_life
commit a3dbbb76cec4d7d2b028c4afef1821262279916a (v1.12.2), author Florance.
Upstream is GPL-3.0; its license is retained in this directory. The containing
Decision plugin is AGPL-3.0; combined redistribution follows AGPL-3.0.

Modifications: original commands and hook registration removed; data moved to
Decision plugin data; Jev approval, native Agent dispatch, exact persona,
WebUI configuration/actions and AIGen photo/video bridge are implemented in
../life_runtime.py and ../life_media.py. No separate Virtual Life plugin is loaded.
Additional integration changes: cancellation-safe atomic storage, rollback on
failed long-term mutations, strict provider selection, and compatibility with
AstrBot versions without ContentPart.mark_as_temp. Upstream Python formatting
was normalized to the containing repository style.

Conversa (https://github.com/Luna-channel/astrbot_plugin_Conversa, MIT) was
reviewed for native CronMessageEvent / result-stage integration and offline
protection. Private Companion (https://github.com/menglimi/astrbot_plugin_private_companion)
was reviewed for expiring candidate / pre-delivery recheck and reference-image
plus daily-outfit composition ideas. Their code is not copied.
