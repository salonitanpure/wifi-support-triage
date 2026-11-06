"""Wi-Fi Support Triage Assistant: a research prototype by Saloni Tanpure.

Reads one customer message sent to a home internet provider and suggests an issue type,
a priority, the key details an agent needs, and a draft reply. A human agent decides.
The model runs on Groq's API; the key is stored in Streamlit secrets, never in this file.
"""
import re, json, time
import requests
import streamlit as st

st.set_page_config(page_title='Wi-Fi Support Triage Assistant', page_icon='📶')

API_URL = 'https://api.groq.com/openai/v1/chat/completions'
MODEL = st.secrets.get('GROQ_MODEL', 'openai/gpt-oss-120b')

CODES = ['outage', 'intermittent', 'slow_speed', 'equipment', 'install_setup',
         'billing_account', 'tv_video', 'unclear', 'other']
NAMES = {
    'outage': 'Outage', 'intermittent': 'Intermittent connection', 'slow_speed': 'Slow speed',
    'equipment': 'Equipment problem', 'install_setup': 'Installation or setup',
    'billing_account': 'Billing or account', 'tv_video': 'TV or video',
    'unclear': 'Unclear, needs more information', 'other': 'Other',
}

SYSTEM = """You triage customer messages sent to a home internet provider's support account.
Reply with JSON only, with these keys:
{"issue_type": "...", "priority": "...", "churn_risk": "yes or no", "location": "...", "steps_tried": "...", "duration": "...", "suggested_reply": "..."}

issue_type (pick exactly one):
- outage: no internet service at all right now
- intermittent: internet drops in and out
- slow_speed: internet works but is slower than expected
- equipment: the router, modem or gateway itself seems faulty
- install_setup: new service, installation, activation or moving, never working yet
- billing_account: price, plan, charges, login, or complaints about the support process
- tv_video: cable TV, channels, DVR, on-demand or the provider's streaming apps
- unclear: about home internet, but the problem type cannot be told
- other: anything else, such as mobile service, home phone, security systems, praise, sales, non-English
If several problems are named, pick the one that blocks service most, in this order: outage, intermittent, equipment, slow_speed, install_setup, billing_account.

priority:
- High: no internet at all; or a problem lasting more than 24 hours without a break; or a problem still unresolved after the customer tried fixes, had a technician visit or contacted support; or any work, health or safety impact.
- Medium: internet works but is degraded with no High signal, including brief recurring drops. install_setup is Medium unless the wait is over a week or work is affected.
- Low: billing_account, tv_video, other, and unclear messages with no High signal.

churn_risk: "yes" if the customer mentions cancelling or switching, names a competitor, or says the problem has lasted months; otherwise "no".
location, steps_tried, duration: copy the customer's own words, or "" if not mentioned.
suggested_reply: one or two short, friendly sentences an agent could send. Never ask the customer to restart equipment they already restarted."""

EXAMPLES = [
    'Is there an outage in the Clearwater, Tampa area? No internet, already reset modem and router.',
    'WiFi keeps cutting out late at night, sick',
    'I live in Texas and my internet, for the past 2 days, has gone from like 44 Mbps and suddenly to a 254 Kbps for quite a while, help?',
    "Getting a replacement modem and it's DOA, after spending 30+ minutes with tech support.",
    'Why will you offer me, a customer for years, internet-only for 70/mo when new customers get 45/mo?',
    'wanna give us our net back? I just got home from a long day and I want to wind down with a show before bed...',
    'Am so done with Comcast. I would never deal with this company again.',
]


def clean(t):
    t = re.sub(r'https?://\S+', '', str(t))
    t = re.sub(r'@\w+', '', t)
    t = t.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return re.sub(r'\s+', ' ', t).strip()


def call_model(msg):
    headers = {'Authorization': f"Bearer {st.secrets['GROQ_API_KEY']}", 'Content-Type': 'application/json'}
    body = {
        'model': MODEL,
        'temperature': 0,
        'max_tokens': 1000,
        'messages': [{'role': 'system', 'content': SYSTEM},
                     {'role': 'user', 'content': f'Message: "{msg}"'}],
    }
    if 'gpt-oss' in MODEL:
        body['reasoning_effort'] = 'low'   # keep the model's hidden thinking short and fast
    r = requests.post(API_URL, headers=headers, json=body, timeout=60)
    if r.status_code == 429:
        raise PermissionError('busy')
    if r.status_code != 200:
        raise RuntimeError(f'HTTP {r.status_code}: {r.text[:300]}')
    return r.json()['choices'][0]['message']['content'] or ''


def parse(raw):
    data = {}
    m = re.search(r'\{.*\}', raw, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
        except Exception:
            data = {}
    issue = str(data.get('issue_type', '')).strip().lower()
    if issue not in CODES:
        hits = [(raw.lower().find(c), c) for c in CODES if re.search(r'\b' + c + r'\b', raw.lower())]
        issue = min(hits)[1] if hits else 'unclear'
    prio = str(data.get('priority', '')).strip().capitalize()
    if prio not in ['High', 'Medium', 'Low']:
        pm = re.search(r'\b(high|medium|low)\b', raw.lower())
        prio = pm.group(1).capitalize() if pm else 'Low'
    churn = 'Yes' if str(data.get('churn_risk', 'no')).strip().lower().startswith('y') else 'No'
    details = {k: str(data.get(k, '') or '').strip() for k in ['location', 'steps_tried', 'duration']}
    reply = str(data.get('suggested_reply', '') or '').strip()
    return issue, prio, churn, details, reply


st.title('Wi-Fi Support Triage Assistant')
st.write(
    'Paste a message a customer sent to their internet provider. The assistant suggests what the problem is, '
    'how urgent it is, the details an agent needs, and a draft reply. It assists a human agent and never replies on its own.'
)
st.caption('Research prototype by Saloni Tanpure, built on public 2017 Twitter support data. '
           "Accuracy results are in the project case study. Don't paste personal information.")

choice = st.selectbox('Start from a real message in the test data, or write your own',
                      ['Write my own'] + EXAMPLES)
idx = 0 if choice == 'Write my own' else EXAMPLES.index(choice) + 1
message = st.text_area('Customer message', value='' if idx == 0 else choice, height=120, key=f'msg_{idx}',
                       placeholder='e.g. My internet has been down since last night and I work from home')

if st.button('Triage message', type='primary'):
    msg = clean(message)
    if not msg:
        st.warning('Paste a customer message first.')
    else:
        try:
            with st.spinner('Reading the message...'):
                start = time.time()
                raw = call_model(msg)
                secs = time.time() - start
        except PermissionError:
            st.error('The free AI service is at its usage limit right now. Try again in a minute.')
            st.stop()
        except Exception as e:
            st.error(f'The AI service returned an error: {e}')
            st.stop()

        issue, prio, churn, d, reply = parse(raw)
        st.subheader(NAMES[issue])
        c1, c2 = st.columns(2)
        c1.metric('Priority', prio)
        c2.metric('Churn risk', churn)
        shown = lambda v: v if v else 'Not mentioned'
        st.markdown(
            '| Detail | From the message |\n| --- | --- |\n'
            f"| Location | {shown(d['location'])} |\n"
            f"| Steps already tried | {shown(d['steps_tried'])} |\n"
            f"| How long | {shown(d['duration'])} |"
        )
        st.markdown('**Draft reply (edit before sending)**')
        st.info(reply or 'No draft generated.')
        st.caption(f'Suggested by {MODEL} in {secs:.1f} s. The agent makes the final call.')
        with st.expander('Raw model output'):
            st.code(raw, language='json')
