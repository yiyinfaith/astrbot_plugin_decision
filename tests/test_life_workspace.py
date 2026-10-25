"""Run the actual browser presentation mapping and prove no setting is lost."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from decision.life_config import life_schema

ROOT = Path(__file__).resolve().parents[1]


def run_js(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable for the frontend mapping test")
    module = ROOT / "pages/dashboard/life-workspace.js"
    result = subprocess.run([node, "-e", "const api=require(" + json.dumps(str(module)) + ");const schema=JSON.parse(require('fs').readFileSync(0,'utf8'));" + script], input=json.dumps(life_schema()), text=True, encoding="utf-8", capture_output=True, check=True)
    return json.loads(result.stdout)


def test_every_original_and_new_setting_has_a_visible_home_without_schema_mutation():
    result = run_js("""
const before=JSON.stringify(schema), missing=[],duplicates=[];
for(const [group,node] of Object.entries(schema)) {
 const counts={};
 for(const panel of api.panels) for(const key of api.keysFor(group,panel.id,schema)) counts[key]=(counts[key]||0)+1;
 // Audience scope controls have dedicated cards in Contact.
 if(['friend_settings','group_settings'].includes(group)) for(const k of ['enable','session_blacklist','session_list','daily_hard_max']) counts[k]=(counts[k]||0)+1;
 for(const k of Object.keys(node.items)) {if(!counts[k]) missing.push(group+'.'+k);if(counts[k]>1) duplicates.push(group+'.'+k);}
 for(const p of api.panels) api.schemaFor(schema,group,p.id);
}
console.log(JSON.stringify({missing,duplicates,unchanged:before===JSON.stringify(schema),panels:api.panels.map(p=>p.id)}));
""")
    assert result == {"missing": [], "duplicates": [], "unchanged": True, "panels": ["contact", "life", "reminders", "media", "status", "advanced"]}


def test_advanced_search_and_common_unit_display_keep_original_values():
    result = run_js("""
const common=api.schemaFor(schema,'adaptive_settings','contact');
const model=api.schemaFor(schema,'schedule_settings','advanced','模型');
const media=api.schemaFor(schema,'media_settings','media');
const settings={enable:true,photo_enable:false,photo_probability:.173};
console.log(JSON.stringify({unit:common.max_interval_minutes.durationUnit,label:common.max_interval_minutes.description,percentage:media.photo_probability.percentage,models:Object.keys(model),hidden:!api.enabledField('media_settings','photo_probability',settings),value:settings.photo_probability}));
""")
    assert result["unit"] == "minutes" and result["percentage"] is True
    assert "（分）" not in result["label"]
    assert result["value"] == .173 and result["hidden"] is True
    assert "schedule_llm_provider" in result["models"] and "proactive_llm_provider" in result["models"]


def test_workspace_has_clear_scope_and_two_distinct_contact_flows():
    page = (ROOT / "pages/dashboard/index.html").read_text("utf-8")
    assert 'aria-label="主动发起功能板块"' in page
    assert "下方设置是全局规则" in page
    assert "话题跟进只作为普通主动候选" in page
    assert "原生 Agent 生成与发送" in page
    assert "按实际发送更新记录" in page
    assert "lifeControl('cancel_task'" in page and "lifeControl('resolve_topic'" in page
