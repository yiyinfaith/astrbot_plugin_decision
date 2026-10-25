/* Presentation only. The persisted configuration keeps every existing key. */
(function (root) {
  const panels = [
    {id:'contact', label:'联系与打扰', description:'联系谁、多久联系、什么时候不打扰，以及谁能调整频率'},
    {id:'life', label:'人格与生活', description:'同一人格的日程、穿搭、外貌、关系和连续生活状态'},
    {id:'reminders', label:'提醒与承诺', description:'明确约定准时执行；未结束话题择机跟进，分别管理'},
    {id:'media', label:'多媒体', description:'照片、视频、音频的工具开关、主动概率、数量与计费'},
    {id:'status', label:'状态与记录', description:'查看所选会话与人格的实际状态、已发内容和未发送原因'},
    {id:'advanced', label:'高级设置', description:'全部精细参数仍可配置；修改前可搜索设置名称'}
  ];
  const common = {
    integration_settings: {contact:['quiet_start','quiet_end']},
    adaptive_settings: {contact:['enable','friend_min_minutes','group_min_minutes','max_interval_minutes','group_adjust_blacklist','group_adjust_users']},
    continuity_settings: {contact:['enable','quality_enabled'],life:['state_enabled'],reminders:['topics_enabled']},
    reply_delay_settings: {contact:['enable','notify_user','max_delay_seconds','active_conversation_seconds']},
    schedule_settings: {life:['daily_schedule_generation_enable','daily_outfit_generation_enable','generate_time']},
    thought_settings: {life:['enable','max_generations_per_persona_day','min_generation_minutes','ttl_minutes']},
    long_term_settings: {life:['enable']},
    smart_context_injection: {life:['enable','base_module_enable']},
    personal_network_integration: {life:['enable','enable_llm_update_tool','enable_llm_query_tool','record_completed_events','inject_current_participants']},
    appearance_settings: {life:['use_reference_outfit']},
    temporal_settings: {reminders:['enable','scheduled_daily_limit','guard_responses','max_lateness_minutes']},
    media_settings: {media:['enable','photo_enable','video_enable','audio_enable','image_aspect_ratio','video_aspect_ratio','video_duration','cooldown_seconds','daily_limit','proactive_billing_mode','proactive_billing_platform','proactive_billing_user_id','photo_probability','video_probability','audio_probability','photo_count_min','photo_count_max','video_count_min','video_count_max','audio_count_min','audio_count_max','audio_mode','audio_voice']}
  };
  const audienceMain = ['enable','session_blacklist','session_list','daily_hard_max'];
  const names = {integration_settings:'免打扰', adaptive_settings:'联系频率与权限',continuity_settings:'对话连续性',reply_delay_settings:'回复节奏',schedule_settings:'每日日程与穿搭',thought_settings:'活动想法与分享',long_term_settings:'长期生活计划',smart_context_injection:'让 Bot 了解当前生活状态',personal_network_integration:'人际关系',appearance_settings:'外貌参考与补充',temporal_settings:'明确提醒与时间校验',media_settings:'照片、视频与音频',creative_pool:'日程素材与风格',prompt_settings:'模型提示词与模板',image_settings:'日程卡片外观',delivery_settings:'主动候选与发送策略',friend_settings:'私聊传统规则',group_settings:'群聊传统规则'};
  function keysFor(key, panel, schema) {
    const all=Object.keys(schema?.[key]?.items || {});
    if (key==='friend_settings' || key==='group_settings') return panel==='advanced' ? all.filter(k=>!audienceMain.includes(k)) : [];
    if (panel==='advanced') {
      const basic = new Set(Object.values(common[key] || {}).flat());
      return all.filter(k=>!basic.has(k));
    }
    return common[key]?.[panel] || [];
  }
  function schemaFor(schema,key,panel,query='') {
    const original=schema?.[key]?.items || {};
    const q=query.trim().toLocaleLowerCase();
    return Object.fromEntries(keysFor(key,panel,schema).filter(k=>k in original && (!q || `${names[key]} ${k} ${original[k].description} ${original[k].hint || ''} ${JSON.stringify(original[k].items || {})}`.toLocaleLowerCase().includes(q))).map(k=>{
      const node={...original[k]};
      if(panel!=='advanced' && /_(minutes|seconds)$/.test(k)) {
        node.durationUnit=k.endsWith('_minutes')?'minutes':'seconds';
        node.description=node.description.replace(/（(?:分|分钟|秒)）|\((?:分|分钟|秒)\)/g,'');
      }
      if(panel==='media' && /_probability$/.test(k)) node.percentage=true;
      return [k,node];
    }));
  }
  function enabledField(group,key,value) {
    if(group==='adaptive_settings' && value.enable===false && ['friend_min_minutes','group_min_minutes','max_interval_minutes'].includes(key)) return false;
    if(group==='continuity_settings' && value.enable===false && key!=='enable') return false;
    if(group==='media_settings') {
      const kind=key.startsWith('photo_') || key==='image_aspect_ratio'?'photo':key.startsWith('video_')?'video':key.startsWith('audio_')?'audio':null;
      if(kind && key!==`${kind}_enable` && value[`${kind}_enable`]===false) return false;
      if(value.enable===false && key!=='enable') return false;
    }
    if(['reply_delay_settings','thought_settings','temporal_settings','long_term_settings','smart_context_injection'].includes(group) && value.enable===false && key!=='enable') return false;
    return true;
  }
  root.LifeWorkspace={panels,common,names,keysFor,schemaFor,enabledField};
  if(typeof module!=='undefined') module.exports=root.LifeWorkspace;
})(globalThis);
