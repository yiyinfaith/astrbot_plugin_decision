const { createApp, nextTick } = Vue;
// The host injects its SDK into the page. AstrBotPluginView is the documented
// name; AstrBotPluginPage remains as an alias on older dashboards.
function currentBridge() { return window.AstrBotPluginView || window.AstrBotPluginPage || null; }
async function waitForBridge(timeoutMs = 5000) {
  const started = Date.now();
  while (!currentBridge() && Date.now() - started < timeoutMs) {
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  const bridge = currentBridge();
  if (!bridge) throw new Error('AstrBot 页面桥接未加载，请重新打开插件页面');
  return bridge;
}
// The liquid-glass theme is optional. While it is off, neither the WebGL
// library nor liquid.css is loaded and the page uses style.css.
const ASSET_VERSION = '0.4.9';
let glassLibraryLoading = null;
let liquidSheetLoading = null;
function supportsWebGL2() {
  try {
    const probe = document.createElement('canvas').getContext('webgl2');
    probe?.getExtension('WEBGL_lose_context')?.loseContext();
    return Boolean(probe);
  } catch (e) { return false; }
}
function loadGlassLibrary() {
  if (window.customElements?.get('liquid-glass')) return Promise.resolve();
  if (!glassLibraryLoading) {
    glassLibraryLoading = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = `./vendor/liquid-glass.js?v=${ASSET_VERSION}`;
      script.onload = resolve;
      script.onerror = () => { glassLibraryLoading = null; script.remove(); reject(new Error('液态玻璃效果加载失败，已保持简洁样式。')); };
      document.head.appendChild(script);
    });
  }
  return glassLibraryLoading;
}
function loadLiquidStylesheet() {
  if (!liquidSheetLoading) {
    liquidSheetLoading = new Promise((resolve, reject) => {
      const link = document.createElement('link');
      link.id = 'liquid-theme';
      link.rel = 'stylesheet';
      link.href = `./liquid.css?v=${ASSET_VERSION}`;
      link.onload = resolve;
      link.onerror = () => { liquidSheetLoading = null; link.remove(); reject(new Error('液态玻璃样式加载失败，已保持简洁样式。')); };
      document.head.appendChild(link);
    });
  }
  return liquidSheetLoading;
}
async function applyLiquidTheme(enabled) {
  const base = document.getElementById('default-theme');
  if (enabled) {
    await Promise.all([loadGlassLibrary(), loadLiquidStylesheet()]);
    if (base) base.disabled = true;
  } else {
    if (base) base.disabled = false;
    document.getElementById('liquid-theme')?.remove();
    liquidSheetLoading = null;
  }
  // The detailed network graph copies colours when it renders.
  window.__decisionNetworkRestyle?.();
}
const DEFAULT_JEV_PROMPT = `你是一个只负责结构化判断的 Decision Model。
请严格根据当前场景、用户请求和候选能力逐项判断。
不要生成最终回答，不要执行工具，不要发明候选名称。`;
const DEFAULT_MAIN_PROMPT = `<decision_routing_hint>
当前场景需要根据用户请求选择合适的能力。
推荐你优先考虑使用以下 Tool：{tools}
如果确有必要调用 MCP 能力，推荐你考虑以下 MCP：{mcps}
如果确有必要使用 Skill，推荐你考虑以下 Skill：{skills}
如果确有必要委派，推荐你考虑以下 SubAgent：{subagents}

以上是 Decision Engine 根据当前场景给出的推荐；你仍需自行判断是否调用、调用哪些参数，以及是否委派。
推荐列表可以为空，也可以同时包含多个候选。
</decision_routing_hint>`;
function payloadOf(value) {
  // The bridge normally removes AstrBot's {status, data} envelope, but
  // older dashboard builds and reverse proxies may leave one or more
  // envelopes in place.  Some deployments also serialize JSON once.
  // Unwrap those forms without treating a valid empty tools array as an
  // error.
  let current = value;
  for (let depth = 0; depth < 6; depth += 1) {
    if (typeof current === 'string') {
      try { current = JSON.parse(current); continue; } catch { return {}; }
    }
    if (!current || typeof current !== 'object') return {};
    if (current.status === 'error' || current.error) throw new Error(current.message || current.error || '服务器返回错误');
    if (Array.isArray(current.tools) || Object.prototype.hasOwnProperty.call(current, 'jev_pre_prompt')) return current;
    if (current.data && typeof current.data === 'object') { current = current.data; continue; }
    if (current.result && typeof current.result === 'object') { current = current.result; continue; }
    return current;
  }
  return current && typeof current === 'object' ? current : {};
}
function normalizeTools(value) {
  const data = payloadOf(value);
  const list = Array.isArray(data.tools) ? data.tools : [];
  return list.filter(item => item && typeof item === 'object' && String(item.name || '').trim()).map(item => ({
    name: String(item.name || '').trim(),
    description: String(item.description || '(无描述)'),
    handoff: item.handoff === true,
    mcp: item.mcp === true,
    skill: item.skill === true,
    active: item.active !== false,
    builtin: item.builtin === true,
    origin_display: String(item.origin_display || (item.builtin ? 'Astrbot内置工具' : '未知来源'))
  }));
}
function hasOwn(value, key) { return Object.prototype.hasOwnProperty.call(value, key); }
const defaults = {
  proactive_whitelist: [], proactive_scope_blacklist: false, direct_reply_prefixes: ['/', '@'], force_reply_when_summoned: true,
  proactive_alias: 'AI|助手', proactive_score_threshold: .68, addressing_threshold: .7,
  interject_threshold: .85, cooldown_seconds: 60, reply_window_seconds: 600,
  max_replies_per_window: 2, proactive_history_max_messages: 12, proactive_max_sessions: 500,
  no_reply_cooldown_seconds: 3, observation_timeout_seconds: 600,
   conversation_flow_analysis_enabled: true, conversation_flow_window: 8, conversation_flow_scope: [], conversation_flow_scope_blacklist: true,
  echo_detection_threshold: 3, echo_detection_window_seconds: 30,
  dense_conversation_threshold: 30, dense_conversation_window_seconds: 600, min_participant_count: 2
};
const TOOL_THRESHOLD_DEFAULT = .65;
const LIFE_PRESETS = {
  restrained: {label:'克制', description:'少打扰：沉默更久才开口，额度更少，对方没回就更快收敛。', values:{
    friend_settings:{idle_min_minutes:180, idle_max_minutes:480, cooldown_minutes:240, daily_budget_min:0, daily_budget_max:2, daily_hard_max:3},
    group_settings:{idle_min_minutes:90, idle_max_minutes:240, cooldown_minutes:120, daily_budget_min:0, daily_budget_max:1, daily_hard_max:1},
    adaptive_settings:{friend_min_minutes:240, group_min_minutes:120, max_interval_minutes:4320},
    delivery_settings:{max_unanswered:2, availability_probabilities:{blocked:0, low:0.1, normal:0.5, high:0.8}},
    integration_settings:{jev_threshold:0.75, jev_relevant_threshold:0.75, jev_timing_threshold:0.6}}},
  normal: {label:'正常', description:'插件默认值：每天一到三次，留出足够的沉默和冷却。', values:{
    friend_settings:{idle_min_minutes:90, idle_max_minutes:240, cooldown_minutes:120, daily_budget_min:1, daily_budget_max:3, daily_hard_max:5},
    group_settings:{idle_min_minutes:45, idle_max_minutes:120, cooldown_minutes:60, daily_budget_min:0, daily_budget_max:1, daily_hard_max:2},
    adaptive_settings:{friend_min_minutes:120, group_min_minutes:60, max_interval_minutes:2880},
    delivery_settings:{max_unanswered:3, availability_probabilities:{blocked:0, low:0.25, normal:0.7, high:1}},
    integration_settings:{jev_threshold:0.7, jev_relevant_threshold:0.7, jev_timing_threshold:0.5}}},
  clingy: {label:'黏人', description:'更常找人：沉默不久就开口，额度更多，但仍遵守免打扰、暂停和每日硬上限。', values:{
    friend_settings:{idle_min_minutes:30, idle_max_minutes:120, cooldown_minutes:45, daily_budget_min:3, daily_budget_max:6, daily_hard_max:10},
    group_settings:{idle_min_minutes:30, idle_max_minutes:90, cooldown_minutes:40, daily_budget_min:1, daily_budget_max:2, daily_hard_max:4},
    adaptive_settings:{friend_min_minutes:45, group_min_minutes:30, max_interval_minutes:1440},
    delivery_settings:{max_unanswered:4, availability_probabilities:{blocked:0, low:0.4, normal:0.9, high:1}},
    integration_settings:{jev_threshold:0.6, jev_relevant_threshold:0.6, jev_timing_threshold:0.4}}}
};
function presetMatches(config, values) {
  return Object.entries(values).every(([group, fields]) => Object.entries(fields).every(([key, value]) => {
    const current = config?.[group]?.[key];
    if (value && typeof value === 'object') return Object.entries(value).every(([k, v]) => Number(current?.[k]) === v);
    return Number(current) === value;
  }));
}
const app = createApp({
  data() {
    return {
      loading: true,
      loaded: false,
      glassReady: false,
      liquidGlass: false,
      providerTouched: false,
      switchingGlass: false,
      saving: false,
      activeTab: 'tools',
      proactiveTab: 'reactive', flowTab: 'input', messageInterjection: true,
      lifePanel: 'contact', lifePanels: LifeWorkspace.panels, lifeAdvancedSearch: '', lifeTaskDraft: {id:'',note:'',runAt:'',originalRunAt:'',originalLocalAt:''}, lifeControlBusy:false,
      lifeConfig: {}, lifeSchema: {}, lifeProviders: [], lifeStatus: {}, lifeTasks: {}, usageStats: {}, lifePresets: LIFE_PRESETS,
      lifeUmo: '', lifeActionText: '', lifeDateRange: '', lifeMessage: '', lifeBusy: false,
      appearancePersona: '', appearanceImages: [], appearanceDescription: '', appearancePersonaId: '', personaOptions: [], sessionOptions: [], sessionSearch: '', personaSearch: '', showSessionList: false, showPersonaList: false,
      networkDetailLoaded: false,
      networkDetailError: '',
      networkPersonaId: '', networkPersonaSearch: '', showNetworkPersonaList: false,
      networkEnabled: false, networkCharacters: [], networkRelations: [], networkEvents: [],
      networkStats: { characters: 0, relationships: 0, events: 0 }, networkBusy: false, networkMessage: '',
      networkNewCharacter: { name: '', bio: '', personality: '' },
      networkNewRelation: { source: 'persona', target: '', type: '', strength: 50 },
      networkNewEvent: { participants: '', type: '', summary: '' },
      billingOptions: [], billingPlatformList: [], billingUserSearch: '', showBillingUserList: false,
      lifeInspection: {}, aigenAvailable: false, lifePoll: null, lifeInspectTask: '', lifeAppliedTask: '',
      lifeToast: null, lifeFreshTask: '',
      toolSearch: '',
      statusText: '加载中…',
      saveState: '',
      tools: [], decisionProviders: [],
      selectedKeep: [],
      selectedRecommend: [],
      selectedKeepSubagents: [],
      selectedRecommendSubagents: [],
      selectedKeepMcp: [],
      selectedRecommendMcp: [],
      selectedKeepSkills: [],
      selectedRecommendSkills: [],
      toolsEnabledValue: true,
      proactiveEnabledValue: false,
      flowEnabledValue: true,
      tabs: [
        {id:'tools', label:'Tool/MCP/Skill/SubAgent 决策', caption:'判断、过滤与能力'},
        {id:'proactive', label:'主动对话', caption:'触发、冷却与范围'},
        {id:'flow', label:'对话流增强', caption:'输入分析与输出处理'}
      ],
      outputStepOptions: [
        {key:'summary',label:'图片外显',hint:'图片消息金句'}, {key:'error',label:'报错处理',hint:'拦截或转发异常'},
        {key:'block',label:'消息拦截',hint:'超时、复读、官腔'}, {key:'at',label:'解析艾特',hint:'假 @ 转真 @'},
        {key:'clean',label:'文本清洗',hint:'清理括号、Emoji'}, {key:'replace',label:'文本替换',hint:'敏感词替换'},
        {key:'typo',label:'错字模拟',hint:'同音错字'}, {key:'tts',label:'文转语音',hint:'TTS 或声聊'},
        {key:'t2i',label:'文转图片',hint:'长文转图'}, {key:'reply',label:'智能引用',hint:'插嘴自动引用'},
        {key:'forward',label:'合并转发',hint:'长消息折叠'}, {key:'recall',label:'自动撤回',hint:'关键词撤回'},
        {key:'split',label:'分段回复',hint:'智能分段与打字'}
      ],
      form: {
        tool_decision_enabled: true,
        mcp_decision_enabled: true,
        skill_decision_enabled: true,
        subagent_decision_enabled: true,
        tool_filter_enabled: true,
        mcp_filter_enabled: true,
        skill_filter_enabled: true,
        subagent_filter_enabled: false,
        tools_scope: '',
        tools_scope_blacklist: true,
        tool_noul_threshold: TOOL_THRESHOLD_DEFAULT,
        decision_provider_id: '',
        jev_pre_prompt: DEFAULT_JEV_PROMPT,
        main_llm_post_prompt: DEFAULT_MAIN_PROMPT,
        proactive: {...defaults},
        dialogue_enhancement: {
          conversation_flow_analysis_enabled: true,
          conversation_flow_window: 8,
          conversation_flow_scope: '',
          conversation_flow_scope_blacklist: true,
          optimize_tool_history_context: false,
          disable_group_at_bot_wake_scope_blacklist: true,
          disable_group_at_bot_wake: false,
          disable_group_at_bot_wake_group_ids: '',
          disable_group_reply_to_bot_wake_scope_blacklist: true,
          disable_group_reply_to_bot_wake: false,
          disable_group_reply_to_bot_wake_group_ids: '',
          multimodal: {models: []},
          output_enhancement_enabled: false,
          output_scope: '',
          output_scope_blacklist: true,
          output_pipeline: {
            pipeline: {lock_order: true, steps: [], llm_steps: []},
            split: {max_length: 200, max_count: 2, per_char_delay: 0.3, char_list: [], delay_scope_str: '1~16', show_typing: true, tail_punc: []},
            reply: {threshold: 1, include_at: false}, forward: {threshold: 150, node_name: ''},
            recall: {delay: 30, keywords: []}, clean: {text_threshold: 150, bracket: true, parenthesis: true, emotion_tag: true, emoji: true, lead: [], tail: [], punctuation: '[#%~]'},
            at: {at_str: true, at_prob: 0.1}, typo: {error_rate: 0.01, tone_error_rate: 0.1, word_replace_rate: 0.01, correction_append_prob: 0.5}, summary: {quotes: [], quotes_files: []},
            error: {keywords: [], custom_msg: '', forward_umo: 'admin'}, block: {timeout: 0, block_reread: true, block_words: []}, t2i: {threshold: 150, pillowmd_style_dir: 'data/plugins/astrbot_plugin_decision/outputboost/t2i_style', auto_page: true, clean_cache: true},
            tts: {group_id: '', character_id: 'lucy-voice-f36', threshold: 50, prob: 0.1, tts_provider_id: ''}, replace: {words: [], default_new_word: '*'}
          }
        }
      }
    };
  },
  computed: {
    // The session picker matters only where data is read per conversation.
    lifeNeedsSession() { return ['life', 'reminders', 'status'].includes(this.lifePanel); },
    lifePresetActive() { return Object.keys(LIFE_PRESETS).find(id => presetMatches(this.lifeConfig, LIFE_PRESETS[id].values)) || ''; },
    lifeAuditRows() { return (this.lifeStatus.audit || []).filter(r => r.umo === this.lifeUmo).slice(-20).reverse(); },
    usageRows() { const day = this.usageStats.days?.[this.usageStats.today] || {}; return Object.entries(day).map(([feature, row]) => ({feature, ...row, tokens: (row.input_tokens || 0) + (row.output_tokens || 0)})).sort((a, b) => b.calls - a.calls); },
    usageWeek() { const days = this.usageStats.days || {}; return Object.entries(days).map(([day, rows]) => ({day, calls: Object.values(rows).reduce((n, r) => n + (r.calls || 0), 0), tokens: Object.values(rows).reduce((n, r) => n + (r.input_tokens || 0) + (r.output_tokens || 0), 0)})); },
    lifeSessions() { return [...new Set([...(this.lifeConfig.friend_settings?.session_list || []), ...(this.lifeConfig.group_settings?.session_list || []), ...Object.keys(this.lifeStatus.sessions || {})])]; },
    // friend/group/network/appearance render as dedicated cards above, so
    // the generic grid must not repeat them.
    lifeConfigGroups() { const special = this.lifePanel === 'life' ? new Set(['personal_network_integration','appearance_settings']) : new Set(); return Object.fromEntries(Object.entries(this.lifeSchema || {}).filter(([key]) => !special.has(key) && Object.keys(this.lifePanelSchema(key)).length)); },
    filteredSessions() { const q = this.sessionSearch.trim().toLowerCase(); const rows = this.sessionOptions || []; if (!q) return rows.slice(0, 60); return rows.filter(row => `${row.title || ''} ${row.umo || ''} ${row.platform || ''} ${row.message_type || ''} ${row.persona_name || ''}`.toLowerCase().includes(q)).slice(0, 60); },
    filteredPersonas() { const q = this.personaSearch.trim().toLowerCase(); const rows = this.personaOptions || []; if (!q) return rows.slice(0, 60); return rows.filter(row => `${row.name || ''} ${row.id || ''}`.toLowerCase().includes(q)).slice(0, 60); },
    filteredNetworkPersonas() { const q = this.networkPersonaSearch.trim().toLowerCase(); const rows = this.personaOptions || []; if (!q) return rows.slice(0, 60); return rows.filter(row => `${row.name || ''} ${row.id || ''}`.toLowerCase().includes(q)).slice(0, 60); },
    filteredBillingAccounts() {
      const q = this.billingUserSearch.trim().toLowerCase();
      const rows = this.billingOptions || [];
      const hit = q ? rows.filter(r => `${r.platform} ${r.user_id} ${r.label || ''}`.toLowerCase().includes(q)) : rows;
      return hit.slice(0, 60);
    },
    // The billing account fields are dropdowns, so they are removed from
    // the generic renderer and drawn explicitly in the media card.
    mediaSettingsSchema() {
      const items = this.lifePanelSchema('media_settings');
      const skip = new Set(['proactive_billing_platform', 'proactive_billing_user_id']);
      return Object.fromEntries(Object.entries(items).filter(([k]) => !skip.has(k)));
    },
    mediaGeneralSchema() { return Object.fromEntries(Object.entries(this.mediaSettingsSchema).filter(([key])=>!this.mediaKindFor(key))); },
    billingPlatforms() { return this.billingPlatformList || []; },
    mm() { return this.form.dialogue_enhancement || {}; },
    mmOcrHint() {
      const tooling = (this.mm.multimodal && this.mm.multimodal.mm_tooling) || {};
      const ocr = tooling.ocr || {};
      const rows = [];
      if (ocr.tesseract && ocr.tesseract.available) rows.push('Tesseract 可用');
      else if (ocr.tesseract && ocr.tesseract.reason) rows.push('Tesseract 不可用：' + ocr.tesseract.reason);
      if (ocr.paddleocr && ocr.paddleocr.available) rows.push('PaddleOCR 可用');
      else if (ocr.paddleocr && ocr.paddleocr.reason) rows.push('PaddleOCR 不可用：' + ocr.paddleocr.reason);
      if (tooling.ffmpeg === false) rows.push('缺少 ffmpeg，视频与音频将不可用');
      return rows.join('；') || '本地 OCR 引擎状态未知';
    },
    mmModels() {
      const models = this.mm.multimodal && this.mm.multimodal.models;
      return Array.isArray(models) ? models : [];
    },
    mmVideoRouteLabel() {
      const config = this.mm.multimodal || {};
      if (config.mm_video_strategy === 'native') return '官方原生：需要 Gemini 官方 Files / Interactions API，反代通常不支持。';
      const id = config.mm_model_video || config.mm_default_model;
      const row = this.mmModels.find(item => item.id === id);
      if (!row) return '请选择视频理解模型。';
      if (config.mm_video_strategy === 'inline') return row.video_inline ? '短片完整视频直传（超过大小预算时改用抽帧）；长片抽帧 + Agentic Seeking。' : '该模型不支持完整视频直传，将按抽帧处理。';
      return '短片自适应抽帧；长片先概览，再由模型自主选择时间段精读。只用 FFmpeg 抽帧与图像理解，兼容各类反代。';
    },
    mmScopeText() {
      const value = this.mm.multimodal && this.mm.multimodal.mm_auto_scope;
      return Array.isArray(value) ? value.join('\n') : String(value || '');
    },
    billingUserLabel() {
      const media = this.lifeConfig?.media_settings;
      if (!media || !media.proactive_billing_user_id) return this.billingUserSearch;
      return media.proactive_billing_platform ? `${media.proactive_billing_platform} · ${media.proactive_billing_user_id}` : media.proactive_billing_user_id;
    },
    toolsEnabled() { return this.toolsEnabledValue; },
    proactiveEnabled() { return this.proactiveEnabledValue; },
    flowEnabled() { return this.flowEnabledValue; },
    outputConfig() { return this.form.dialogue_enhancement.output_pipeline || {}; },
    visibleTools() { const q = this.toolSearch.trim().toLowerCase(); return this.tools.filter(t => !q || `${t.name} ${t.origin_display} ${t.description}`.toLowerCase().includes(q)); },
    ordinaryVisibleTools() { return this.visibleTools.filter(t => !t.handoff && !t.mcp && !t.skill); },
    mcpVisibleTools() { return this.visibleTools.filter(t => t.mcp && !t.handoff); },
    skillVisibleTools() { return this.visibleTools.filter(t => t.skill && !t.handoff && !t.mcp); },
    subagentVisibleTools() { return this.visibleTools.filter(t => t.handoff); },
    groupedTools() { const map = new Map(); for (const t of this.ordinaryVisibleTools) { const key = t.origin_display || '未知来源'; if (!map.has(key)) map.set(key, []); map.get(key).push(t); } return [...map.entries()].map(([origin,items]) => ({origin,items})); },
    groupedMcp() { const map = new Map(); for (const t of this.mcpVisibleTools) { const key = t.origin_display || '未知来源'; if (!map.has(key)) map.set(key, []); map.get(key).push(t); } return [...map.entries()].map(([origin,items]) => ({origin,items})); },
    groupedSkills() { const map = new Map(); for (const t of this.skillVisibleTools) { const key = t.origin_display || 'Skill'; if (!map.has(key)) map.set(key, []); map.get(key).push(t); } return [...map.entries()].map(([origin,items]) => ({origin,items})); },
    groupedSubagents() { const map = new Map(); for (const t of this.subagentVisibleTools) { const key = t.origin_display || '未知来源'; if (!map.has(key)) map.set(key, []); map.get(key).push(t); } return [...map.entries()].map(([origin,items]) => ({origin,items})); }
  },
  methods: {
    cleanLifeLabel(text) { return String(text || '').replace(/^[^\p{L}\p{N}]+/u, ''); },
    async refreshOptions() {
      try {
        const bridge = await waitForBridge();
        const [rawPersonas, rawSessions] = await Promise.all([bridge.apiGet('life/personas'), bridge.apiGet('life/sessions')]);
        const personas = payloadOf(rawPersonas).personas; const sessions = payloadOf(rawSessions).sessions;
        if (Array.isArray(personas)) this.personaOptions = personas.filter(row => row && row.id).map(row => ({id:String(row.id), name:String(row.name || row.id)}));
        if (Array.isArray(sessions)) this.sessionOptions = sessions.filter(row => row && row.umo).map(row => ({umo:String(row.umo), title:String(row.title || ''), platform:String(row.platform || ''), message_type:String(row.message_type || ''), persona_id:String(row.persona_id || ''), persona_name:String(row.persona_name || '')}));
      } catch(e) { this.lifeMessage = `会话/人格列表读取失败：${e.message || e}（仍可直接粘贴 UMO 或人格 ID）`; }
    },
    mediaKindFor(key) { return key==='image_aspect_ratio' || key.startsWith('photo_')?'photo':key.startsWith('video_')?'video':key.startsWith('audio_')?'audio':null; },
    mediaKindSchema(kind) { return Object.fromEntries(Object.entries(this.mediaSettingsSchema).filter(([key])=>this.mediaKindFor(key)===kind)); },
    lifePanelSchema(key) { const schema=LifeWorkspace.schemaFor(this.lifeSchema,key,this.lifePanel,this.lifePanel==='advanced'?this.lifeAdvancedSearch:''); if(this.lifePanel==='advanced') return schema; const value=this.lifeConfig[key] || {}; return Object.fromEntries(Object.entries(schema).filter(([field])=>LifeWorkspace.enabledField(key,field,value))); },
    lifeGroupTitle(key) { return LifeWorkspace.names[key] || this.cleanLifeLabel(this.lifeSchema[key]?.description || key); },
    lifeGroupSummary(key) { const v=this.lifeConfig[key] || {}; if(key==='adaptive_settings') return v.enable?'自适应已开启':'使用传统规则'; if(key==='humanization_settings') return v.enable?'关系、节奏与内容轮换已开启':'拟人化优化已关闭'; if(key==='integration_settings') return `${v.quiet_start}—${v.quiet_end} · 阈值 ${v.jev_threshold}/${v.jev_relevant_threshold}/${v.jev_timing_threshold}`; if(key==='weather_settings') return ({off:'不使用天气',random:'随机天气',city:'城市天气 · ' + (v.city || '天气插件默认城市')})[v.mode] || '不使用天气'; if(key==='media_settings') return ['photo','video','audio'].filter(k=>v[k+'_enable']).map(k=>({photo:'照片',video:'视频',audio:'音频'}[k])).join(' · ') || '多媒体已关闭'; if('enable' in v) return v.enable?'已开启':'已关闭'; return '全局规则'; },
    localTaskTime(value) { const d=new Date(value); return Number.isNaN(d.getTime())?'':new Date(d.getTime()-d.getTimezoneOffset()*60000).toISOString().slice(0,19); },
    async lifeControl(action,extra={}) { if(!this.lifeUmo.trim()) return this.showToast('err','未选择会话','先选择目标会话'); const umo=this.lifeUmo.trim(); this.lifeControlBusy=true; try { const bridge=await waitForBridge(); const result=payloadOf(await bridge.apiPost('life/control',{umo,action,...extra})); if(!result.saved) throw new Error(result.error || '操作未保存'); if(this.lifeUmo.trim()===umo) { await this.lifeAction('inspect'); this.showToast('ok','已保存','任务与状态已更新'); } return true; } catch(e) {this.showToast('err','操作未保存',e.message || String(e));} finally {this.lifeControlBusy=false;} },
    async createLifeTask() { if(!this.lifeTaskDraft.note || !this.lifeTaskDraft.runAt) return this.showToast('err','信息不完整','填写提醒内容和时间'); const draft={...this.lifeTaskDraft}; const runAt=draft.id && draft.runAt===draft.originalLocalAt ? draft.originalRunAt : new Date(draft.runAt).toISOString(); if(await this.lifeControl(draft.id?'edit_task':'create_task',{task_id:draft.id,note:draft.note,run_at:runAt})) this.lifeTaskDraft={id:'',note:'',runAt:'',originalRunAt:'',originalLocalAt:''}; },
    editLifeTask(task) { const local=this.localTaskTime(task.run_at); this.lifeTaskDraft={id:task.id,note:task.note,runAt:local,originalRunAt:task.run_at,originalLocalAt:local}; this.$nextTick(()=>document.getElementById('life-task-editor')?.scrollIntoView({behavior:'smooth',block:'center'})); },
    listText(group, key) { const value = this.lifeConfig?.[group]?.[key]; return Array.isArray(value) ? value.join('\n') : String(value || ''); },
    setScopeList(group, key, raw) { if (!this.lifeConfig[group]) return; this.lifeConfig[group][key] = String(raw || '').split('\n').map(item => item.trim()).filter(Boolean); },
    onNetworkDetailToggle(event) {
      // Boot the ported network view on first open, then leave its own
      // state alone so collapsing the card never loses the graph.
      if (!event.target.open) return;
      if (this.networkDetailLoaded) {
        // cytoscape measured the canvas while the card was collapsed, so
        // tell it to re-measure now that it is visible again.
        if (typeof window.__decisionNetworkResize === 'function') {
          window.setTimeout(() => window.__decisionNetworkResize(), 0);
        }
        return;
      }
      this.loadNetworkDetail();
    },
    async loadNetworkDetail() {
      if (this.networkDetailLoaded || this._networkLoading) return;
      this._networkLoading = true;
      try {
        // The panel must be visible before cytoscape measures its canvas,
        // otherwise the graph initialises to a zero-sized viewport.
        this.networkDetailLoaded = true;
        await this.$nextTick();
        // cytoscape is a classic script; the view itself is a module.
        if (!window.cytoscape) {
          await this.loadScript(`./vendor/cytoscape.min.js?v=${ASSET_VERSION}`);
        }
        await import(`./network-app.js?v=${ASSET_VERSION}`);
      } catch (e) {
        this.networkDetailLoaded = false;
        this._networkLoading = false;
        this.networkDetailError = `关系图加载失败：${e.message || e}`;
      }
    },
    loadScript(src) {
      return new Promise((resolve, reject) => {
        const tag = document.createElement('script');
        tag.src = src;
        tag.onload = () => resolve();
        tag.onerror = () => reject(new Error(`无法加载 ${src}`));
        document.head.appendChild(tag);
      });
    },
    setMmScope(raw) {
      if (!this.mm.multimodal) return;
      this.mm.multimodal.mm_auto_scope = String(raw || '')
        .split(/\n|[,，]/).map(item => item.trim()).filter(Boolean);
    },
    async reloadModels() {
      try {
        const bridge = await waitForBridge();
        const result = payloadOf(await bridge.apiGet('multimodal/models'));
        if (result && Array.isArray(result.models) && this.mm.multimodal) {
          this.mm.multimodal.models = result.models;
          this.statusText = `已重新加载 ${result.models.length} 个可用模型`;
        }
      } catch (e) {
        this.statusText = `模型列表加载失败：${e.message || e}`;
      }
    },
    toggleScope(group) { const settings = this.lifeConfig?.[group]; if (settings) settings.session_blacklist = !settings.session_blacklist; },
    hidePicker(kind) { if (kind === 'session') this.showSessionList = false; else if (kind === 'billingUser') this.showBillingUserList = false; else if (kind === 'networkPersona') this.showNetworkPersonaList = false; else this.showPersonaList = false; },
    onBillingUserInput(raw) { this.billingUserSearch = String(raw || ''); this.showBillingUserList = true; },
    pickBillingAccount(row) {
      this.showBillingUserList = false;
      if (!this.lifeConfig.media_settings) this.lifeConfig.media_settings = {};
      this.lifeConfig.media_settings.proactive_billing_user_id = row.user_id;
      if (row.platform) this.lifeConfig.media_settings.proactive_billing_platform = row.platform;
      this.billingUserSearch = '';
      this.lifeMessage = `主动发起将记到 ${row.platform} / ${row.user_id}（余额 ${row.amount}）。`;
    },
    onSessionInput(raw) {
      const value = String(raw || ''); this.sessionSearch = value; this.showSessionList = true;
      const trimmed = value.trim();
      const resolved = this.sessionOptions.some(row => row.umo === trimmed)
        ? trimmed : (/\S+:\S+:\S+/.test(trimmed) ? trimmed : '');
      const changed = resolved && resolved !== this.lifeUmo;
      this.lifeUmo = resolved;
      if (changed) this.loadLifeForSession();
    },
    pickSession(row) { this.showSessionList = false; this.sessionSearch = row.title || row.umo; this.lifeUmo = row.umo; this.loadLifeForSession(); },
    async pickPersona(row) { this.showPersonaList = false; this.personaSearch = row.name || row.id; this.appearancePersonaId = row.id; await this.loadAppearance(row.id); if (!this.networkPersonaId) { this.networkPersonaId = row.id; this.networkPersonaSearch = row.id; this.reloadNetwork(); } },
    pickNetworkPersona(row) {
      this.showNetworkPersonaList = false;
      this.networkPersonaSearch = row.name || row.id;
      this.networkPersonaId = row.id;
      this.reloadNetwork();
    },
    networkNameOf(id) {
      const row = (this.networkCharacters || []).find(c => c.id === id);
      return row ? row.name : (id || '').slice(0, 8);
    },
    async reloadNetwork() {
      if (!this.networkPersonaId) return;
      this.networkBusy = true;
      try {
        const bridge = await waitForBridge();
        const result = payloadOf(await bridge.apiGet('network/network', { persona_id: this.networkPersonaId }));
        this.networkCharacters = Array.isArray(result.characters) ? result.characters : [];
        this.networkRelations = Array.isArray(result.relationships) ? result.relationships : [];
        this.networkEvents = Array.isArray(result.life_events) ? result.life_events : [];
        this.networkEnabled = !!(result.network && result.network.enabled);
        this.networkStats = {
          characters: this.networkCharacters.filter(c => !c.is_persona).length,
          relationships: this.networkRelations.length,
          events: this.networkEvents.length,
        };
        this.networkMessage = '';
      } catch (e) {
        this.networkMessage = `读取人际网络失败：${e.message || e}`;
      } finally { this.networkBusy = false; }
    },
    async toggleNetworkEnabled(value) {
      if (!this.networkPersonaId) return;
      this.networkBusy = true;
      try {
        const bridge = await waitForBridge();
        await bridge.apiPost('network/network/enabled', { persona_id: this.networkPersonaId, enabled: value === true });
        this.networkEnabled = value === true;
        this.networkMessage = value ? '已启用该人格的人际网络。' : '已停用该人格的人际网络。';
      } catch (e) { this.networkMessage = `切换失败：${e.message || e}`; }
      finally { this.networkBusy = false; }
    },
    async addNetworkCharacter() {
      const form = this.networkNewCharacter;
      if (!this.networkPersonaId || !form.name.trim()) return;
      this.networkBusy = true;
      try {
        const bridge = await waitForBridge();
        await bridge.apiPost('network/character/save', {
          persona_id: this.networkPersonaId, name: form.name.trim(),
          bio: form.bio || '', personality: form.personality || '',
          alias_usages: [], preferences: [], facts: [], notes: '',
        });
        this.networkNewCharacter = { name: '', bio: '', personality: '' };
        this.networkMessage = '人物已保存。';
        await this.reloadNetwork();
      } catch (e) { this.networkMessage = `保存人物失败：${e.message || e}`; }
      finally { this.networkBusy = false; }
    },
    async removeNetworkCharacter(row) {
      if (!this.networkPersonaId || row.is_persona) return;
      this.networkBusy = true;
      try {
        const bridge = await waitForBridge();
        await bridge.apiPost('network/character/delete', { persona_id: this.networkPersonaId, character_id: row.id });
        this.networkMessage = '人物已删除。';
        await this.reloadNetwork();
      } catch (e) { this.networkMessage = `删除失败：${e.message || e}`; }
      finally { this.networkBusy = false; }
    },
    async addNetworkRelation() {
      const form = this.networkNewRelation;
      if (!this.networkPersonaId || !form.target.trim() || !form.type.trim()) return;
      this.networkBusy = true;
      try {
        const bridge = await waitForBridge();
        await bridge.apiPost('network/relationship/save', {
          persona_id: this.networkPersonaId, source: form.source || 'persona',
          target: form.target.trim(), type: form.type.trim(),
          strength: Number(form.strength) || 0, status: 'active', description: '',
        });
        this.networkNewRelation = { source: 'persona', target: '', type: '', strength: 50 };
        this.networkMessage = '关系已保存。';
        await this.reloadNetwork();
      } catch (e) { this.networkMessage = `保存关系失败：${e.message || e}`; }
      finally { this.networkBusy = false; }
    },
    async removeNetworkRelation(row) {
      if (!this.networkPersonaId) return;
      this.networkBusy = true;
      try {
        const bridge = await waitForBridge();
        await bridge.apiPost('network/relationship/delete', { persona_id: this.networkPersonaId, relationship_id: row.id });
        this.networkMessage = '关系已删除。';
        await this.reloadNetwork();
      } catch (e) { this.networkMessage = `删除失败：${e.message || e}`; }
      finally { this.networkBusy = false; }
    },
    async addNetworkEvent() {
      const form = this.networkNewEvent;
      if (!this.networkPersonaId || !form.summary.trim()) return;
      this.networkBusy = true;
      try {
        const bridge = await waitForBridge();
        const names = String(form.participants || '').split(/[,，、]/).map(s => s.trim()).filter(Boolean);
        await bridge.apiPost('network/life-event/save', {
          persona_id: this.networkPersonaId,
          occurred_at: new Date().toISOString(),
          type: form.type || '共同经历',
          participants: names,
          summary: form.summary.trim(),
          importance: 50, emotional_tone: '',
        });
        this.networkNewEvent = { participants: '', type: '', summary: '' };
        this.networkMessage = '经历已保存。';
        await this.reloadNetwork();
      } catch (e) { this.networkMessage = `保存经历失败：${e.message || e}`; }
      finally { this.networkBusy = false; }
    },
    async removeNetworkEvent(row) {
      if (!this.networkPersonaId) return;
      this.networkBusy = true;
      try {
        const bridge = await waitForBridge();
        await bridge.apiPost('network/life-event/delete', { persona_id: this.networkPersonaId, event_id: row.id });
        this.networkMessage = '经历已删除。';
        await this.reloadNetwork();
      } catch (e) { this.networkMessage = `删除失败：${e.message || e}`; }
      finally { this.networkBusy = false; }
    },
    async loadBillingOptions() {
      try {
        const bridge = await waitForBridge();
        const result = payloadOf(await bridge.apiGet('life/billing'));
        this.billingOptions = Array.isArray(result.accounts) ? result.accounts : [];
        this.billingPlatformList = Array.isArray(result.platforms) ? result.platforms : [];
      } catch (e) { this.lifeMessage = `计费账户列表读取失败：${e.message || e}`; }
    },
    async loadAppearance(personaId) {
      const target = String(personaId || '').trim();
      if (!target) { this.appearanceImages = []; this.appearanceDescription = ''; return; }
      this.lifeBusy = true;
      try {
        const bridge = await waitForBridge(); const result = payloadOf(await bridge.apiPost('life/appearance',{action:'list', persona_id:target}));
        this.appearancePersonaId = result.persona_id || target; this.appearanceImages = result.images || []; this.appearanceDescription = String(result.description || '');
      } catch(e) { this.appearanceImages = []; this.lifeMessage = `读取外貌参考图失败：${e.message || e}`; }
      finally { this.lifeBusy = false; }
    },
    async refreshLife() {
      try {
        const bridge = await waitForBridge(); const result = payloadOf(await bridge.apiGet('life'));
        this.lifeStatus = result.status || {}; this.lifeTasks = result.tasks || {}; this.aigenAvailable = result.aigen_available === true; this.usageStats = result.usage || {};
        const running = Object.values(this.lifeTasks).some(task => task.state === 'running');
        if (this.lifePoll) clearTimeout(this.lifePoll);
        if (running) this.lifePoll = setTimeout(() => this.refreshLife(), 2000);
      } catch(e) { this.lifeMessage = `读取后台状态失败：${e.message || e}`; }
      await this.refreshOptions();
      await this.loadBillingOptions();
    },
    async loadLifeForSession() {
      const umo = String(this.lifeUmo || '').trim();
      if (!umo) return;
      // Selecting a conversation is enough; never require a separate click.
      if (this._lifeLoading === umo) return;
      this._lifeLoading = umo;
      this.lifeBusy = true;
      this.lifeMessage = `正在读取 ${umo} 的人格与今日生活…`;
      this.showToast('busy', '正在读取', umo);
      try {
        const bridge = await waitForBridge();
        const result = payloadOf(await bridge.apiPost('life/appearance', { umo, action: 'list' }));
        if (String(this.lifeUmo || '').trim() !== umo) return;
        this.appearancePersona = result.persona_id;
        this.appearancePersonaId = result.persona_id || this.appearancePersonaId;
        this.appearanceImages = result.images || [];
        this.appearanceDescription = String(result.description || '');
        await this.lifeAction('inspect');
      } catch (e) {
        this.lifeMessage = `读取失败：${e.message || e}`;
      } finally {
        if (this._lifeLoading === umo) this._lifeLoading = '';
        this.lifeBusy = false;
      }
    },
    actionLabel(action) {
      return ({
        inspect: '读取生活与外貌',
        generate: '生成今日内容',
        rewrite_schedule: '重写日程',
        rewrite_outfit: '重写穿搭',
        render_schedule: '渲染日程图片',
        render_outfit: '渲染穿搭图片',
        render_timeline: '渲染长期阶段图片',
        render_stage: '查询当前阶段',
        initiate: '请求立即发起',
        settle_delay: '结算回复等待',
        generate_timeline: '生成追加草稿',
        regenerate_timeline: '生成替换草稿',
        import_timeline: '导入 JSON',
        approve_timeline: '批准草稿',
        reject_timeline: '拒绝并重写',
        add_requirement: '添加一次性要求',
      })[action] || action;
    },
    showToast(kind, title, text) { this.lifeToast = { kind, title, text }; },
    applyLifePreset(id) { const preset = LIFE_PRESETS[id]; if (!preset) return; for (const [group, fields] of Object.entries(preset.values)) { if (!this.lifeConfig[group]) continue; for (const [key, value] of Object.entries(fields)) { if (value && typeof value === 'object') this.lifeConfig[group][key] = {...(this.lifeConfig[group][key] || {}), ...value}; else this.lifeConfig[group][key] = value; } } this.showToast('success', `已套用「${preset.label}」`, '只改了频率相关的参数，保存后生效；之后仍可逐项微调。'); },
    formatTokens(value) { const n = Number(value) || 0; return n >= 10000 ? (n / 10000).toFixed(1) + ' 万' : String(n); },
    describeResult(result) {
      if (result == null) return '完成，没有返回内容。';
      if (typeof result === 'string') return result.slice(0, 300);
      if (Array.isArray(result)) {
        const texts = result.filter(p => p && p.type === 'text').map(p => String(p.text || ''));
        const images = result.filter(p => p && p.type === 'image').length;
        const head = texts.join(' ').replace(/\s+/g, ' ').slice(0, 240);
        const tail = images ? ` （含 ${images} 张图片，见下方结果）` : '';
        return (head || '完成。') + tail;
      }
      if (typeof result === 'object') {
        if (typeof result.message === 'string') return result.message.slice(0, 300);
        if (result.sent === true) return '已发送。';
        if (result.sent === false) return `未发送：${result.reason || '未知原因'}`;
        return JSON.stringify(result).slice(0, 240);
      }
      return String(result).slice(0, 240);
    },
    refreshStatus(q) {
      if (q === 'running') return '处理中…';
      if (q === 'done') return '已完成';
      if (q === 'error') return '失败';
      if (q === 'cancelled') return '已取消';
      return q;
    },
    async lifeAction(action) {
      if (!this.lifeUmo.trim()) { this.lifeMessage = '请先填写完整会话 UMO。'; return; }
      this.lifeBusy = true;
      try {
        const bridge = await waitForBridge(); const result = payloadOf(await bridge.apiPost('life/action',{umo:this.lifeUmo.trim(), action, text:this.lifeActionText, date_range:this.lifeDateRange}));
        if (!result.task_id) throw new Error('服务器未返回任务编号');
        this.lifeMessage = '正在处理…';
        if (action === 'inspect') this.lifeInspectTask = result.task_id;
        // Poll until this specific job settles, otherwise every button
        // looks dead because the result only appears later.
        const done = await this.awaitLifeTask(result.task_id);
        await this.refreshLife();
        const label = this.actionLabel(action);
        const job = (this.lifeTasks || {})[result.task_id] || {};
        if (!done) {
          this.lifeMessage = `操作仍在后台运行（${label}）；稍后点“刷新状态”查看结果。`;
          this.showToast('busy', label, '仍在后台运行，可稍后刷新查看结果。');
        } else if (job.error) {
          this.lifeMessage = `操作失败：${job.error}`;
          this.showToast('err', `${label} 失败`, job.error);
        } else {
          this.lifeMessage = `已完成：${label}`;
          this.lifeFreshTask = result.task_id;
          this.showToast('ok', `${label} 已完成`, this.describeResult(job.result));
        }
      } catch(e) {
        this.lifeMessage = `操作失败：${e.message || e}`;
        this.showToast('err', `${this.actionLabel(action)} 失败`, String(e.message || e));
      }
      finally { this.lifeBusy = false; }
    },
    async awaitLifeTask(taskId, timeoutMs = 120000) {
      const bridge = await waitForBridge();
      const started = Date.now();
      while (Date.now() - started < timeoutMs) {
        await new Promise(resolve => setTimeout(resolve, 700));
        try {
          const result = payloadOf(await bridge.apiGet('life'));
          this.lifeStatus = result.status || this.lifeStatus;
          this.lifeTasks = result.tasks || this.lifeTasks;
          const job = (this.lifeTasks || {})[taskId];
          if (job && job.state !== 'running') return true;
        } catch (e) { /* keep polling through a transient error */ }
      }
      return false;
    },
    async uploadAppearance(event) {
      const file = event.target.files?.[0]; if (!file) return;
      if (!this.appearancePersonaId) { this.lifeMessage = '请先选择人格，参考图按人格保存。'; event.target.value = ''; return; }
      if (file.size > 8*1024*1024) { this.lifeMessage = '参考图不能超过 8MB。'; event.target.value = ''; return; }
      this.lifeBusy = true;
      try {
        const data = await new Promise((resolve,reject)=>{ const reader=new FileReader(); reader.onload=()=>resolve(reader.result); reader.onerror=()=>reject(new Error('文件读取失败')); reader.readAsDataURL(file); });
        const bridge = await waitForBridge(); const result = payloadOf(await bridge.apiPost('life/appearance',{persona_id:this.appearancePersonaId,action:'upload',name:file.name,data_url:data}));
        this.appearancePersonaId = result.persona_id || this.appearancePersonaId; this.appearanceImages = result.images || []; this.lifeMessage = '外貌参考图已保存到 AstrBot 数据目录。';
      } catch(e) { this.lifeMessage = `上传失败：${e.message || e}`; }
      finally { this.lifeBusy = false; event.target.value = ''; }
    },
    async removeAppearance(id) {
      if (!this.appearancePersonaId) { this.lifeMessage = '请先选择人格。'; return; }
      this.lifeBusy = true;
      try { const bridge=await waitForBridge(); const result=payloadOf(await bridge.apiPost('life/appearance',{persona_id:this.appearancePersonaId,action:'delete',id})); this.appearanceImages=result.images || []; }
      catch(e) { this.lifeMessage=`删除失败：${e.message || e}`; }
      finally { this.lifeBusy = false; }
    },
    async saveAppearanceDescription() {
      if (!this.appearancePersonaId) return;
      try {
        const bridge = await waitForBridge();
        await bridge.apiPost('life/appearance', {persona_id:this.appearancePersonaId, action:'set_description', description:this.appearanceDescription});
        this.lifeMessage = '当前人格的外貌补充已保存。';
      } catch(e) { this.lifeMessage = `外貌补充保存失败：${e.message || e}`; }
    },
    outputStepEnabled(key) { const list = this.outputConfig?.pipeline?.steps || []; return list.some(item => String(item).split('(', 1)[0].trim() === key); },
    toggleOutputStep(key) { const pipeline = this.form.dialogue_enhancement.output_pipeline.pipeline; const list = Array.isArray(pipeline.steps) ? [...pipeline.steps] : []; const index = list.findIndex(item => String(item).split('(', 1)[0].trim() === key); if (index >= 0) list.splice(index, 1); else { const label = this.outputStepOptions.find(item => item.key === key)?.label || key; list.push(`${key}(${label})`); } pipeline.steps = list; },
    outputLlmStepEnabled(key) { const list = this.outputConfig?.pipeline?.llm_steps || []; return list.some(item => String(item).split('(', 1)[0].trim() === key); },
    toggleOutputLlmStep(key) { const pipeline = this.form.dialogue_enhancement.output_pipeline.pipeline; const list = Array.isArray(pipeline.llm_steps) ? [...pipeline.llm_steps] : []; const index = list.findIndex(item => String(item).split('(', 1)[0].trim() === key); if (index >= 0) list.splice(index, 1); else { const label = this.outputStepOptions.find(item => item.key === key)?.label || key; list.push(`${key}(${label})`); } pipeline.llm_steps = list; },
    outputList(group, key) { const value = this.outputConfig?.[group]?.[key]; return Array.isArray(value) ? value.join('\n') : String(value || ''); },
    setOutputList(group, key, raw) { if (!this.form.dialogue_enhancement.output_pipeline[group]) this.form.dialogue_enhancement.output_pipeline[group] = {}; this.form.dialogue_enhancement.output_pipeline[group][key] = String(raw || '').split(/\n|[,，]/).map(item => item.trim()).filter(Boolean); },
    scopeList(value) { return String(value || '').split(/\n|[,，]/).map(item => item.trim()).filter(Boolean); },
    syncThresholdGlass() {
      const el = document.getElementById('thresholdGlass');
      if (!this.glassReady || !el?.setState) return;
      el.dataset.syncing = '1';
      try { el.setState({sliderValue: Math.round(Math.max(0, Math.min(1, Number(this.form.tool_noul_threshold) || 0)) * 100)}); }
      finally { delete el.dataset.syncing; }
    },
    syncDecisionRing() {
      const el = document.getElementById('decisionRing');
      if (!this.glassReady || !el?.setState) return;
      el.dataset.syncing = '1';
      try { el.setState({ringProgressValue: Math.round(Math.max(0, Math.min(1, Number(this.form.tool_noul_threshold) || 0)) * 100)}); }
      finally { delete el.dataset.syncing; }
    },
    syncGlassNav() {
      const el = document.getElementById('glassNav');
      if (!this.glassReady || !el?.setState) return;
      el.dataset.syncing = '1';
      try { el.setState({selectedTab: this.tabs.findIndex(tab => tab.id === this.activeTab)}); }
      finally { delete el.dataset.syncing; }
    },
    async applySavedGlass(enabled) {
      // Display preference only; any failure quietly keeps the default theme.
      if (enabled !== true || !supportsWebGL2()) return;
      await this.setLiquidGlass(true).catch(() => this.setLiquidGlass(false).catch(() => {}));
    },
    async setLiquidGlass(value) {
      // Serialised: the switch is disabled while a change is in progress.
      if (this.switchingGlass) throw new Error('样式正在切换，请稍候');
      this.switchingGlass = true;
      try {
        if (!value) this.releaseGlass();
        await applyLiquidTheme(value);
        this.liquidGlass = value;
        if (value) {
          await nextTick();
          await this.initGlass().catch(() => { this.glassReady = false; });
        }
      } finally {
        this.switchingGlass = false;
      }
    },
    releaseGlass() {
      this.glassReady = false;
      if (this._navMedia && this._navLabelChange) this._navMedia.removeEventListener('change', this._navLabelChange);
      this._navMedia = null; this._navLabelChange = null;
    },
    async toggleLiquidGlass(target) {
      const value = target.checked;
      try {
        if (this.switchingGlass || value === this.liquidGlass) return;
        if (value && !supportsWebGL2()) throw new Error('当前浏览器不支持 WebGL2，无法显示液态玻璃效果');
        const previous = this.liquidGlass;
        try {
          await this.setLiquidGlass(value);
          const bridge = await waitForBridge();
          const result = payloadOf(await bridge.apiPost('ui-settings/save', { ui_liquid_glass: value }));
          if (result.saved !== true) throw new Error('服务器未确认保存');
          this.statusText = value ? '已开启液态玻璃效果' : '已切换为简洁样式';
          this.saveState = 'ok';
        } catch (e) {
          await this.setLiquidGlass(previous).catch(() => {});
          throw e;
        }
      } catch (e) {
        this.statusText = `切换样式失败：${e.message || e}`;
        this.saveState = 'error';
      } finally {
        // :checked is a one-way binding; keep the control in step with the real state.
        target.checked = this.liquidGlass;
      }
    },
    async initGlass() {
      if (!window.customElements || !this.liquidGlass || !supportsWebGL2()) return;
      const registered = await Promise.race([
        customElements.whenDefined('liquid-glass').then(() => true),
        new Promise(resolve => setTimeout(() => resolve(false), 8000))
      ]);
      if (!registered || !this.liquidGlass) return;
      const nav = document.getElementById('glassNav');
      const slider = document.getElementById('thresholdGlass');
      const ring = document.getElementById('decisionRing');
      const ordinarySelect = document.getElementById('ordinarySelectGlass');
      const ordinaryClear = document.getElementById('ordinaryClearGlass');
      const mcpSelect = document.getElementById('mcpSelectGlass');
      const mcpClear = document.getElementById('mcpClearGlass');
      const subagentSelect = document.getElementById('subagentSelectGlass');
      const subagentClear = document.getElementById('subagentClearGlass');
      const skillSelect = document.getElementById('skillSelectGlass');
      const skillClear = document.getElementById('skillClearGlass');
      if (!nav?.setTabs || !slider?.setState || !ring?.setState || !ordinarySelect?.setButtons || !ordinaryClear?.setButtons || !mcpSelect?.setButtons || !mcpClear?.setButtons || !subagentSelect?.setButtons || !subagentClear?.setButtons || !skillSelect?.setButtons || !skillClear?.setButtons) return;
      if (this._navMedia && this._navLabelChange) this._navMedia.removeEventListener('change', this._navLabelChange);
      const navMedia = window.matchMedia('(max-width:600px)');
      const updateNavLabels = () => { nav.setTabs([[
        {label:navMedia.matches ? '能力决策' : 'Tool/MCP/Skill/SubAgent 决策', icon:'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z', viewport:24},
        {label:'主动对话', icon:'M20 2H4a2 2 0 0 0-2 2v18l4-4h14a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2z', viewport:24},
        {label:'对话流增强', icon:'M22.7 19l-9.1-9.1a6 6 0 0 0-7.7-7.7l3.8 3.8-3.7 3.7-3.8-3.8a6 6 0 0 0 7.7 7.7l9.1 9.1z', viewport:24}
      ]]); this.syncGlassNav(); };
      this._navMedia=navMedia; this._navLabelChange=updateNavLabels;
      navMedia.addEventListener('change',updateNavLabels);
      updateNavLabels();
      nav.addEventListener('lg-statechange', e => {
        const index = e.detail?.selectedTab;
        if (!nav.dataset.syncing && Number.isInteger(index) && this.tabs[index]) this.activeTab = this.tabs[index].id;
      });
      slider.addEventListener('lg-statechange', e => {
        const value = e.detail?.sliderValue;
        if (!slider.dataset.syncing && Number.isFinite(value)) this.form.tool_noul_threshold = Math.round(Math.max(0, Math.min(100, value))) / 100;
      });
      ring.addEventListener('lg-statechange', e => {
        const value = e.detail?.ringProgressValue;
        if (!ring.dataset.syncing && Number.isFinite(value)) this.form.tool_noul_threshold = Math.round(Math.max(0, Math.min(100, value))) / 100;
      });
      // Goose rasterizes button labels into a canvas. Keep its liquid
      // shape as the backdrop, but put the actual labels in native HTML
      // buttons so Chinese glyphs stay sharp on every screen density.
      // NBSPs make each hidden Goose shape fill its half-row container, so
      // the rows must be visible before they are measured.
      this.glassReady = true;
      await nextTick();
      const blankGlassLabel = el => {
        const measure = document.createElement('canvas').getContext('2d');
        if (measure) measure.font = '400 15px -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Arial, sans-serif';
        const spaceWidth = Math.max(1, measure?.measureText('\u00a0').width || 4);
        const targetWidth = Math.max(120, el.clientWidth - 10);
        const count = Math.max(1, Math.ceil((targetWidth - 32) / spaceWidth));
        return '\u00a0'.repeat(count);
      };
      ordinarySelect.setButtons([{id:'select-tools', label:blankGlassLabel(ordinarySelect), style:'blue'}]);
      ordinaryClear.setButtons([{id:'clear-tools', label:blankGlassLabel(ordinaryClear), style:'purple'}]);
      mcpSelect.setButtons([{id:'select-mcp', label:blankGlassLabel(mcpSelect), style:'blue'}]);
      mcpClear.setButtons([{id:'clear-mcp', label:blankGlassLabel(mcpClear), style:'purple'}]);
      subagentSelect.setButtons([{id:'select-subagents', label:blankGlassLabel(subagentSelect), style:'blue'}]);
      subagentClear.setButtons([{id:'clear-subagents', label:blankGlassLabel(subagentClear), style:'purple'}]);
      skillSelect.setButtons([{id:'select-skills', label:blankGlassLabel(skillSelect), style:'blue'}]);
      skillClear.setButtons([{id:'clear-skills', label:blankGlassLabel(skillClear), style:'purple'}]);
      this.syncThresholdGlass();
      this.syncDecisionRing();
      this.syncGlassNav();
    },
    keepList(tool) { return tool.handoff ? this.selectedKeepSubagents : (tool.mcp ? this.selectedKeepMcp : (tool.skill ? this.selectedKeepSkills : this.selectedKeep)); },
    recommendList(tool) { return tool.handoff ? this.selectedRecommendSubagents : (tool.mcp ? this.selectedRecommendMcp : (tool.skill ? this.selectedRecommendSkills : this.selectedRecommend)); },
    isKept(name, handoff = false, mcp = false, skill = false) { return (handoff ? this.selectedKeepSubagents : (mcp ? this.selectedKeepMcp : (skill ? this.selectedKeepSkills : this.selectedKeep))).includes(name); },
    isRecommended(name, handoff = false, mcp = false, skill = false) { return (handoff ? this.selectedRecommendSubagents : (mcp ? this.selectedRecommendMcp : (skill ? this.selectedRecommendSkills : this.selectedRecommend))).includes(name); },
    toggleKeep(tool) { const list = this.keepList(tool); const recommended = this.recommendList(tool); const kept = list.includes(tool.name); const next = tool.active && !kept ? [...list, tool.name] : list.filter(n => n !== tool.name); if (tool.handoff) { this.selectedKeepSubagents = next; if (kept) this.selectedRecommendSubagents = recommended.filter(n => n !== tool.name); } else if (tool.mcp) { this.selectedKeepMcp = next; if (kept) this.selectedRecommendMcp = recommended.filter(n => n !== tool.name); } else if (tool.skill) { this.selectedKeepSkills = next; if (kept) this.selectedRecommendSkills = recommended.filter(n => n !== tool.name); } else { this.selectedKeep = next; if (kept) this.selectedRecommend = recommended.filter(n => n !== tool.name); } },
    toggleRecommend(tool) { const list = this.recommendList(tool); const next = list.includes(tool.name) ? list.filter(n => n !== tool.name) : [...list, tool.name]; if (tool.handoff) this.selectedRecommendSubagents = next; else if (tool.mcp) this.selectedRecommendMcp = next; else if (tool.skill) this.selectedRecommendSkills = next; else this.selectedRecommend = next; },
    selectVisibleTools() { for (const tool of this.ordinaryVisibleTools) if (tool.active && !this.isKept(tool.name, false)) this.selectedKeep.push(tool.name); this.selectedKeep = [...new Set(this.selectedKeep)]; },
    clearVisibleTools() { const names = new Set(this.ordinaryVisibleTools.map(t => t.name)); this.selectedKeep = this.selectedKeep.filter(n => !names.has(n)); this.selectedRecommend = this.selectedRecommend.filter(n => !names.has(n)); },
    selectVisibleMcp() { for (const tool of this.mcpVisibleTools) if (tool.active && !this.isKept(tool.name, false, true)) this.selectedKeepMcp.push(tool.name); this.selectedKeepMcp = [...new Set(this.selectedKeepMcp)]; },
    clearVisibleMcp() { const names = new Set(this.mcpVisibleTools.map(t => t.name)); this.selectedKeepMcp = this.selectedKeepMcp.filter(n => !names.has(n)); this.selectedRecommendMcp = this.selectedRecommendMcp.filter(n => !names.has(n)); },
    selectVisibleSkills() { for (const tool of this.skillVisibleTools) if (tool.active && !this.isKept(tool.name, false, false, true)) this.selectedKeepSkills.push(tool.name); this.selectedKeepSkills = [...new Set(this.selectedKeepSkills)]; },
    clearVisibleSkills() { const names = new Set(this.skillVisibleTools.map(t => t.name)); this.selectedKeepSkills = this.selectedKeepSkills.filter(n => !names.has(n)); this.selectedRecommendSkills = this.selectedRecommendSkills.filter(n => !names.has(n)); },
    selectVisibleSubagents() { for (const tool of this.subagentVisibleTools) if (tool.active && !this.isKept(tool.name, true)) this.selectedKeepSubagents.push(tool.name); this.selectedKeepSubagents = [...new Set(this.selectedKeepSubagents)]; },
    clearVisibleSubagents() { const names = new Set(this.subagentVisibleTools.map(t => t.name)); this.selectedKeepSubagents = this.selectedKeepSubagents.filter(n => !names.has(n)); this.selectedRecommendSubagents = this.selectedRecommendSubagents.filter(n => !names.has(n)); },
    async save() {
      if (!this.loaded) { this.statusText = '尚未成功加载，已阻止保存'; this.saveState = 'error'; return; }
      this.saving = true; this.saveState = ''; this.statusText = '保存中…';
      try {
        const activeBridge = await waitForBridge();
        const numberOr = (value, fallback) => { const n = Number(value); return Number.isFinite(n) ? n : fallback; };
        const proactive = {...this.form.proactive, proactive_whitelist: this.scopeList(this.form.proactive.proactive_whitelist), direct_reply_prefixes: this.scopeList(this.form.proactive.direct_reply_prefixes)};
        for (const key of Object.keys(defaults)) if (typeof defaults[key] === 'number') proactive[key] = numberOr(proactive[key], defaults[key]);
        const dialogue = JSON.parse(JSON.stringify(this.form.dialogue_enhancement));
        dialogue.conversation_flow_scope = this.scopeList(dialogue.conversation_flow_scope);
        dialogue.disable_group_at_bot_wake_group_ids = this.scopeList(dialogue.disable_group_at_bot_wake_group_ids);
        dialogue.disable_group_reply_to_bot_wake_group_ids = this.scopeList(dialogue.disable_group_reply_to_bot_wake_group_ids);
        dialogue.output_scope = this.scopeList(dialogue.output_scope);
        const body = {life: JSON.parse(JSON.stringify(this.lifeConfig)), message_interjection_enabled: this.messageInterjection, always_keep_tools: Array.from(this.selectedKeep, String), always_keep_recommend_tools: Array.from(this.selectedRecommend, String), always_keep_mcp: Array.from(this.selectedKeepMcp, String), always_keep_recommend_mcp: Array.from(this.selectedRecommendMcp, String), always_keep_skills: Array.from(this.selectedKeepSkills, String), always_keep_recommend_skills: Array.from(this.selectedRecommendSkills, String), always_keep_subagents: Array.from(this.selectedKeepSubagents, String), always_keep_recommend_subagents: Array.from(this.selectedRecommendSubagents, String), jev_pre_prompt: String(this.form.jev_pre_prompt ?? ''), main_llm_post_prompt: String(this.form.main_llm_post_prompt ?? ''), tool_decision_enabled: this.form.tool_decision_enabled === true, mcp_decision_enabled: this.form.mcp_decision_enabled === true, skill_decision_enabled: this.form.skill_decision_enabled === true, subagent_decision_enabled: this.form.subagent_decision_enabled === true, tool_filter_enabled: this.form.tool_filter_enabled === true, mcp_filter_enabled: this.form.mcp_filter_enabled === true, skill_filter_enabled: this.form.skill_filter_enabled === true, subagent_filter_enabled: this.form.subagent_filter_enabled === true, tools_scope: this.scopeList(this.form.tools_scope), tools_scope_blacklist: this.form.tools_scope_blacklist === true, tool_noul_threshold: numberOr(this.form.tool_noul_threshold, TOOL_THRESHOLD_DEFAULT), proactive, dialogue_enhancement: dialogue};
        // Only send the Provider when it was picked here, so a page opened
        // before a change on AstrBot's native config page cannot undo it.
        const sentProvider = String(this.form.decision_provider_id || '');
        if (this.providerTouched) body.decision_provider_id = sentProvider;
        const result = payloadOf(await activeBridge.apiPost('settings/save', JSON.parse(JSON.stringify(body))));
        if (result.saved !== true) throw new Error('服务器未确认保存成功');
        if (Array.isArray(result.decision_providers)) this.decisionProviders = result.decision_providers;
        // A choice made while this save was in flight stays pending for the next save.
        if (String(this.form.decision_provider_id || '') === sentProvider) {
          if (typeof result.decision_provider_id === 'string') this.form.decision_provider_id = result.decision_provider_id;
          this.providerTouched = false;
        }
        if (Array.isArray(result.always_keep_tools)) this.selectedKeep = result.always_keep_tools.map(String);
        if (Array.isArray(result.always_keep_recommend_tools)) this.selectedRecommend = result.always_keep_recommend_tools.map(String);
        if (Array.isArray(result.always_keep_mcp)) this.selectedKeepMcp = result.always_keep_mcp.map(String);
        if (Array.isArray(result.always_keep_recommend_mcp)) this.selectedRecommendMcp = result.always_keep_recommend_mcp.map(String);
        if (Array.isArray(result.always_keep_skills)) this.selectedKeepSkills = result.always_keep_skills.map(String);
        if (Array.isArray(result.always_keep_recommend_skills)) this.selectedRecommendSkills = result.always_keep_recommend_skills.map(String);
        if (Array.isArray(result.always_keep_subagents)) this.selectedKeepSubagents = result.always_keep_subagents.map(String);
        if (Array.isArray(result.always_keep_recommend_subagents)) this.selectedRecommendSubagents = result.always_keep_recommend_subagents.map(String);
        this.statusText = result.runtime_warning ? `已保存，后台启动失败：${result.runtime_warning}` : '已保存'; this.saveState = result.runtime_warning ? 'error' : 'ok'; this.refreshLife();
      } catch (e) { this.statusText = `保存失败：${e.message || e}`; this.saveState = 'error'; }
      finally { this.saving = false; }
    },
    async load() {
      try {
        const activeBridge = await waitForBridge();
        // Follow the dashboard's light/dark theme (style.css reads html[data-theme]).
        const applyTheme = context => { if (typeof context?.isDark === 'boolean') document.documentElement.dataset.theme = context.isDark ? 'dark' : 'light'; };
        applyTheme(await activeBridge.ready());
        this._offContext = activeBridge.onContext?.(applyTheme);
        const [rawToolData, rawSettings, rawLife] = await Promise.all([activeBridge.apiGet('tools'), activeBridge.apiGet('settings'), activeBridge.apiGet('life')]);
        const life = payloadOf(rawLife); if (!life.schema || !life.settings) throw new Error('生活设置未完整加载，已阻止保存');
        this.lifeSchema = life.schema; this.lifeConfig = life.settings; this.lifeProviders = life.providers || []; this.lifeStatus = life.status || {}; this.lifeTasks = life.tasks || {}; this.aigenAvailable = life.aigen_available === true; this.usageStats = life.usage || {};
        const toolData = payloadOf(rawToolData); const settings = payloadOf(rawSettings);
        if (!Array.isArray(toolData.tools) || !settings || typeof settings !== 'object' || Array.isArray(settings) || !Array.isArray(settings.always_keep_tools)) throw new Error('服务器返回的数据格式无效');
        this.tools = normalizeTools(toolData);
        this.decisionProviders = Array.isArray(settings.decision_providers) ? settings.decision_providers : [];
        this.form.decision_provider_id = String(settings.decision_provider_id || '');
        this.providerTouched = false;
        this.messageInterjection = settings.message_interjection_enabled !== false;
        this.selectedKeep = settings.always_keep_tools.map(String);
        this.selectedRecommend = Array.isArray(settings.always_keep_recommend_tools) ? settings.always_keep_recommend_tools.map(String) : [];
        this.selectedKeepMcp = Array.isArray(settings.always_keep_mcp) ? settings.always_keep_mcp.map(String) : [];
        this.selectedRecommendMcp = Array.isArray(settings.always_keep_recommend_mcp) ? settings.always_keep_recommend_mcp.map(String) : [];
        this.selectedKeepSkills = Array.isArray(settings.always_keep_skills) ? settings.always_keep_skills.map(String) : [];
        this.selectedRecommendSkills = Array.isArray(settings.always_keep_recommend_skills) ? settings.always_keep_recommend_skills.map(String) : [];
        this.selectedKeepSubagents = Array.isArray(settings.always_keep_subagents) ? settings.always_keep_subagents.map(String) : [];
        this.selectedRecommendSubagents = Array.isArray(settings.always_keep_recommend_subagents) ? settings.always_keep_recommend_subagents.map(String) : [];
        this.form.tool_decision_enabled = settings.tool_decision_enabled !== false; this.form.mcp_decision_enabled = settings.mcp_decision_enabled !== false; this.form.skill_decision_enabled = settings.skill_decision_enabled !== false; this.form.subagent_decision_enabled = settings.subagent_decision_enabled !== false; this.form.tool_filter_enabled = settings.tool_filter_enabled !== false; this.form.mcp_filter_enabled = settings.mcp_filter_enabled !== false; this.form.skill_filter_enabled = settings.skill_filter_enabled !== false; this.form.subagent_filter_enabled = settings.subagent_filter_enabled === true; this.form.tools_scope = (Array.isArray(settings.tools_scope) ? settings.tools_scope : []).map(String).join('\n'); this.form.tools_scope_blacklist = settings.tools_scope_blacklist !== false; this.form.tool_noul_threshold = Number.isFinite(Number(settings.tool_noul_threshold)) ? Number(settings.tool_noul_threshold) : TOOL_THRESHOLD_DEFAULT;
        this.form.jev_pre_prompt = hasOwn(settings, 'jev_pre_prompt') ? String(settings.jev_pre_prompt ?? '') : DEFAULT_JEV_PROMPT; this.form.main_llm_post_prompt = hasOwn(settings, 'main_llm_post_prompt') ? String(settings.main_llm_post_prompt ?? '') : DEFAULT_MAIN_PROMPT;
        this.form.proactive = {...defaults, ...(settings.proactive || {})}; this.form.proactive.proactive_whitelist = (settings.proactive?.proactive_whitelist || []).join('\n'); this.form.proactive.direct_reply_prefixes = (settings.proactive?.direct_reply_prefixes || ['/', '@']).join(', ');
        const incoming = settings.dialogue_enhancement || {}; this.form.dialogue_enhancement = {...this.form.dialogue_enhancement, ...incoming}; this.form.dialogue_enhancement.conversation_flow_scope = (incoming.conversation_flow_scope || []).join('\n'); this.form.dialogue_enhancement.disable_group_at_bot_wake_group_ids = (incoming.disable_group_at_bot_wake_group_ids || []).join('\n'); this.form.dialogue_enhancement.disable_group_reply_to_bot_wake_group_ids = (incoming.disable_group_reply_to_bot_wake_group_ids || []).join('\n'); this.form.dialogue_enhancement.output_scope = (incoming.output_scope || []).join('\n'); if (incoming.output_pipeline) this.form.dialogue_enhancement.output_pipeline = incoming.output_pipeline;
        this.toolsEnabledValue = settings.tools_subagents_decision_enabled !== false; this.proactiveEnabledValue = settings.proactive_reply_enabled === true; this.flowEnabledValue = settings.conversation_flow_enabled !== false; this.loaded = true; this.statusText = `已加载 ${this.tools.length} 个能力`; this.syncThresholdGlass();
        this.applySavedGlass(settings.ui_liquid_glass);
        await this.refreshOptions();
        await this.loadBillingOptions();
        const personaId = String(this.appearancePersonaId || this.lifeConfig?.appearance_settings?.persona_id || '').trim();
        if (personaId) {
          this.appearancePersonaId = personaId;
          this.personaSearch = personaId;
          await this.loadAppearance(personaId);
          // The network panel is always visible, so seed it with the same
          // persona instead of waiting for the user to pick one again.
          this.networkPersonaId = personaId;
          this.networkPersonaSearch = personaId;
          await this.reloadNetwork();
        }
      } catch (e) { this.statusText = `加载失败：${e.message || e}`; this.saveState = 'error'; }
      finally { this.loading = false; }
    }
  },
  watch: {
    lifeUmo() { this.lifeTaskDraft={id:'',note:'',runAt:'',originalRunAt:'',originalLocalAt:''}; this.lifeInspection={}; this.appearanceImages=[]; this.appearanceDescription=''; this.appearancePersona=''; this.appearancePersonaId=''; this.personaSearch=''; this.lifeInspectTask=''; this.lifeAppliedTask=''; },
    lifeTasks: {deep:true, handler(tasks) { for (const [id,task] of Object.entries(tasks).reverse()) { if (task.umo !== this.lifeUmo.trim() || task.state !== 'done') continue; const inspection=task.inspection || (task.action === 'inspect' ? task.result : null); if (inspection) { if (id !== this.lifeAppliedTask) { this.lifeAppliedTask=id; this.lifeInspection=inspection; this.appearancePersona=inspection.persona_id || this.appearancePersona; if (inspection.persona_id && inspection.persona_id !== this.appearancePersonaId) { this.appearancePersonaId=inspection.persona_id; this.personaSearch=inspection.persona_id; this.loadAppearance(inspection.persona_id); } } break; } } }},
    'form.tool_noul_threshold'() { this.syncThresholdGlass(); this.syncDecisionRing(); },
    activeTab() { this.syncGlassNav(); }
  },
  beforeUnmount() { if (this.lifePoll) clearTimeout(this.lifePoll); this.releaseGlass(); if (typeof this._offContext === 'function') this._offContext(); },
  mounted() { this.load(); }
});
app.component('life-duration', {
  props:['modelValue','base','label'], emits:['update:modelValue'],
  data() { const m=this.base==='seconds'?Number(this.modelValue)/60:Number(this.modelValue); return {unit:m>=1440 && m%1440===0?'days':m>=60 && m%60===0?'hours':this.base}; },
  computed:{scale(){return ({seconds:1,minutes:60,hours:3600,days:86400}[this.unit])/(this.base==='seconds'?1:60);}, display(){return Number((Number(this.modelValue)/this.scale).toFixed(6));}},
  template:`<div class="life-duration"><input type="number" min="0" step="any" :aria-label="label" :value="display" @input="$emit('update:modelValue',Number($event.target.value)*scale)"><select :aria-label="label+'单位'" v-model="unit"><option v-if="base==='seconds'" value="seconds">秒</option><option value="minutes">分钟</option><option value="hours">小时</option><option value="days">天</option></select></div>`
});
app.component('life-settings', {
  name: 'LifeSettings', props: ['schema','value','providers'],
  methods: {
    label(text) { return String(text || '').replace(/^[^\p{L}\p{N}]+/u, ''); },
    set(key,node,raw) { this.value[key] = node.type === 'list' ? String(raw).split('\n').map(s=>s.trim()).filter(Boolean) : ['int','float'].includes(node.type) ? Number(raw) : raw; }
  },
  template: `<div class="life-setting-fields"><template v-for="(node,key) in schema" :key="key"><details v-if="node.type === 'object'"><summary>{{ label(node.description || key) }}</summary><life-settings :schema="node.items" :value="value[key]" :providers="providers"></life-settings></details><div v-else-if="node.type === 'bool'" class="switch-row"><div class="switch-copy"><strong>{{ label(node.description || key) }}</strong><small v-if="node.hint">{{ node.hint }}</small></div><label class="switch"><input type="checkbox" :aria-label="label(node.description || key)" v-model="value[key]"><span class="switch-track"></span></label></div><div v-else class="field"><label>{{ label(node.description || key) }}</label><life-duration v-if="node.durationUnit" :model-value="value[key]" :base="node.durationUnit" :label="label(node.description || key)" @update:model-value="value[key]=$event"></life-duration><div v-else-if="node.percentage" class="life-percent"><input type="number" min="0" max="100" step="0.1" :aria-label="label(node.description || key)" :value="Number((Number(value[key])*100).toFixed(6))" @input="value[key]=Number($event.target.value)/100"><span>%</span></div><select :aria-label="label(node.description || key)" v-else-if="key.includes('llm_provider')" v-model="value[key]"><option value="">{{ key === 'schedule_llm_provider' ? '沿用 AstrBot 默认模型' : '沿用当前会话模型' }}</option><option v-for="provider in providers" :key="provider.id" :value="provider.id">{{ provider.name || provider.id }}</option><option v-if="value[key] && !providers.some(p=>p.id === value[key])" :value="value[key]">{{ value[key] }}</option></select><select :aria-label="label(node.description || key)" v-else-if="node.options" v-model="value[key]"><option v-for="(option,idx) in node.options" :key="option" :value="option">{{ node.labels?.[idx] || option }}</option></select><textarea :aria-label="label(node.description || key)" v-else-if="node.type === 'list'" :value="(value[key] || []).join(String.fromCharCode(10))" @input="set(key,node,$event.target.value)" class="list-editor" placeholder="每行一个"></textarea><textarea :aria-label="label(node.description || key)" v-else-if="node.type === 'text' || String(value[key] || '').length > 120" :value="value[key]" @input="set(key,node,$event.target.value)" spellcheck="false"></textarea><input v-else :aria-label="label(node.description || key)" :type="['int','float'].includes(node.type) ? 'number' : 'text'" :step="node.type === 'float' ? '0.01' : '1'" :value="value[key]" @input="set(key,node,$event.target.value)"><p v-if="node.hint" class="hint">{{ node.hint }}</p></div></template></div>`
});
// <liquid-glass> is a Web Component provided by the optional vendor library.
app.config.compilerOptions.isCustomElement = tag => tag === 'liquid-glass';
app.mount('#app');
