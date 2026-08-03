import atexit
import difflib
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
import streamlit as st

# Streamlit execs page scripts directly, so sys.path isn't guaranteed to contain
# apps/tools-app (the parent of this package) regardless of which entrypoint launched it.
_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from wiz.job_app_backend import MODEL_DIR, predict, highlight_html, LABEL_COLOURS, load_model, skill_llm
from job_app.storage import FILENAMES, SessionStorage, bucket_from_env, client_from_env

_BACKEND_SPACY = "spaCy NER"
_BACKEND_LLM = f"LLM ({skill_llm.DEFAULT_MODEL})"


@st.cache_resource
def _load_skill_model():
    return load_model()


@st.cache_data
def _predict_skills_spacy(text: str) -> list[dict]:
    _load_skill_model()
    return predict(text)


@st.cache_data(ttl=300)
def _predict_skills_llm(text: str) -> list[dict]:
    return skill_llm.predict(text)


@st.cache_data(ttl=30)
def _llm_available() -> bool:
    return skill_llm.is_available()


_BACKEND_DISPATCH = {
    _BACKEND_SPACY: _predict_skills_spacy,
    _BACKEND_LLM: _predict_skills_llm,
}

_PERSONAL_FIELDS = {"name", "phone", "city", "email", "address", "linkedin", "github", "website", "mobile"}
_CLAUDE = shutil.which("claude") or "/opt/homebrew/bin/claude"
_MAX_HISTORY_TURNS = 10
_NEW_SESSION = "— new session —"
_NEW_VERSION = "— new version —"


@st.cache_resource
def _get_storage() -> SessionStorage:
    return SessionStorage(client_from_env(), bucket_from_env())


def _anonymize_cv(text: str) -> str:
    def redact(m: re.Match) -> str:
        field = m.group(1).lower()
        if field in _PERSONAL_FIELDS:
            return f"\\def\\{m.group(1)}{{REDACTED}}"
        return m.group(0)
    return re.sub(r"\\def\\([a-zA-Z]+)\{([^}]*)\}", redact, text)


def _anonymize_application(text: str) -> str:
    text = re.sub(r"(?<!\d)(\+\d{1,3}[\s\-]?)?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)", "REDACTED", text)
    text = re.sub(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "REDACTED", text)
    return text


def _mirror_dir() -> Path:
    """A per-browser-session local scratch directory the Claude CLI can read/write against.

    The Claude CLI only speaks the filesystem, so session files are mirrored down from MinIO
    into this directory before each call and any write Claude makes is uploaded back after.
    """
    if "_mirror_dir" not in st.session_state:
        mirror = tempfile.mkdtemp(prefix="job_app_mirror_")
        atexit.register(shutil.rmtree, mirror, ignore_errors=True)
        st.session_state["_mirror_dir"] = mirror
    return Path(st.session_state["_mirror_dir"])


def _run_claude(
    storage: SessionStorage, mirror: Path, prompt: str, output_target: tuple[str, str, str] | None = None
) -> str:
    storage.materialize(mirror)
    output_path = mirror.joinpath(*output_target) if output_target else None
    write_instruction = (
        f"When writing files, ONLY write to {output_path}. "
        "Never modify the files you read from — treat them as read-only source material."
        if output_path else
        "Do not write or modify any files unless explicitly asked."
    )
    system = (
        "You are a job application assistant. "
        f"The working directory is {mirror}, which contains session subfolders "
        "each with version subdirectories (e.g. draft/, final/) containing cv.tex, application.tex, and job_posting.tex. "
        "Help the user craft, review, and improve their job applications. "
        + write_instruction
    )
    result = subprocess.run(
        [_CLAUDE, "-p", prompt, "--system-prompt", system, "--add-dir", str(mirror),
         "--allowedTools", "Edit", "Write", "Read", "Bash"],
        cwd=mirror,
        capture_output=True,
        text=True,
        timeout=120,
    )
    reply = (result.stdout or result.stderr).strip()
    if output_target is not None and output_path is not None and output_path.exists():
        storage.save(*output_target, output_path.read_text(encoding="utf-8"))
    return reply


_QUICK_PROMPTS = {
    "": "— or pick a quick action —",
    "📝 Review CV / cover letter": (
        "Review the CV and cover letter in the current session for clarity, impact, and professionalism. "
        "Give specific, actionable feedback."
    ),
    "🎯 Align with job posting": (
        "Compare the CV and cover letter against the job posting. "
        "How well do they align? What should be emphasised more?"
    ),
    "✍️ Rewrite / improve sections": (
        "Identify the weakest sections in the CV and cover letter and suggest concrete rewrites."
    ),
    "🔍 Identify gaps": (
        "What gaps exist between the candidate's profile (CV) and the job requirements? "
        "Be specific about what is missing or underdeveloped."
    ),
    "📊 Suggest keywords": (
        "Extract the most important keywords and phrases from the job posting "
        "that should appear in the CV and cover letter but currently do not."
    ),
}

_FILE_LABELS = {"cv.tex": "CV", "application.tex": "Application Letter", "job_posting.tex": "Job Posting"}


def _session_version_label(session: str, version: str) -> str:
    return f"{session} / {version}"


# --- Page ---

st.title("Job Application")

try:
    storage = _get_storage()
    sessions = storage.list_sessions()
    _session_versions = {s: storage.list_versions(s) for s in sessions}
except Exception as exc:
    st.error(
        f"Could not reach MinIO storage: {exc}\n\n"
        "Start it with `docker compose up -d` from `apps/tools-app/` (see `.env.example`), then reload."
    )
    st.stop()

col_sess, col_ver, col_name = st.columns([2, 2, 3])
with col_sess:
    selected_session = st.selectbox(
        "Session",
        options=[_NEW_SESSION] + sessions,
        index=0,
    )
with col_ver:
    if selected_session != _NEW_SESSION:
        versions = _session_versions[selected_session]
        selected_version = st.selectbox(
            "Version",
            options=[_NEW_VERSION] + versions,
            index=1 if versions else 0,
        )
    else:
        versions = []
        selected_version = _NEW_VERSION
        st.selectbox("Version", [_NEW_VERSION], disabled=True)
with col_name:
    name_cols = st.columns(2)
    with name_cols[0]:
        default_sess_name = "" if selected_session == _NEW_SESSION else selected_session
        session_name = st.text_input("Session name", value=default_sess_name, placeholder="e.g. company-role")
    with name_cols[1]:
        default_ver_name = "" if selected_version == _NEW_VERSION else selected_version
        version_name = st.text_input("Version name", value=default_ver_name, placeholder="e.g. draft, final")

# Load files once when selected session/version changes
_load_key = (selected_session, selected_version)
if st.session_state.get("_last_load_key") != _load_key:
    st.session_state["_last_load_key"] = _load_key
    if selected_session != _NEW_SESSION and selected_version != _NEW_VERSION:
        st.session_state["cv_text"] = storage.load(selected_session, selected_version, "cv.tex")
        st.session_state["application_text"] = storage.load(selected_session, selected_version, "application.tex")
        st.session_state["job_text"] = storage.load(selected_session, selected_version, "job_posting.tex")
        st.session_state["saved_cv"] = st.session_state["cv_text"]
        st.session_state["saved_application"] = st.session_state["application_text"]
        st.session_state["saved_job"] = st.session_state["job_text"]
    else:
        for key in ("cv_text", "application_text", "job_text", "saved_cv", "saved_application", "saved_job"):
            st.session_state[key] = ""
    # Sync shell defaults to match top-level selection
    _all_sv_now = [(s, v) for s in sessions for v in _session_versions[s]]
    if selected_session != _NEW_SESSION and selected_version != _NEW_VERSION:
        try:
            st.session_state["in_sv"] = _all_sv_now.index((selected_session, selected_version))
        except ValueError:
            st.session_state["in_sv"] = -1
    else:
        st.session_state["in_sv"] = -1
    st.session_state["out_sv"] = _NEW_VERSION
    st.session_state["out_new_sess"] = selected_session if selected_session != _NEW_SESSION else ""
    st.session_state["out_new_ver"] = "final"

st.divider()

tab_editor, tab_viewer, tab_shell = st.tabs(["✏️ Editor", "🔍 Viewer", "🖥️ Shell"])

# Helper: build a flat list of (session, version) pairs for pickers
_ALL_SV = [(s, v) for s in sessions for v in _session_versions[s]]
_SV_LABELS = [_session_version_label(s, v) for s, v in _ALL_SV]

with tab_editor:
    if sessions:
        with st.expander("Copy content from existing session"):
            copy_cols = st.columns([3, 1, 1, 1])
            with copy_cols[0]:
                copy_idx = st.selectbox("Source", range(len(_ALL_SV)), format_func=lambda i: _SV_LABELS[i], key="copy_src", label_visibility="collapsed")
            with copy_cols[1]:
                copy_cv = st.checkbox("CV", value=True)
            with copy_cols[2]:
                copy_app = st.checkbox("Application Letter")
            with copy_cols[3]:
                copy_job = st.checkbox("Job Posting")
            if st.button("Copy"):
                src_s, src_v = _ALL_SV[copy_idx]
                if copy_cv:
                    st.session_state["cv_text"] = storage.load(src_s, src_v, "cv.tex")
                if copy_app:
                    st.session_state["application_text"] = storage.load(src_s, src_v, "application.tex")
                if copy_job:
                    st.session_state["job_text"] = storage.load(src_s, src_v, "job_posting.tex")
                st.rerun()

    st.subheader("CV")
    cv = st.text_area(
        label="cv", key="cv_text", height=500,
        placeholder="Paste your LaTeX CV here…", label_visibility="collapsed",
    )

    st.divider()

    st.subheader("Job Posting")
    job_posting = st.text_area(
        label="job posting", key="job_text", height=300,
        placeholder="Paste the job posting (LaTeX or plain text) here…", label_visibility="collapsed",
    )

    st.divider()

    st.subheader("Application Letter")
    application = st.text_area(
        label="application", key="application_text", height=400,
        placeholder="Paste your LaTeX application letter here…", label_visibility="collapsed",
    )

    st.divider()

    can_save = session_name.strip() and version_name.strip()
    if st.button("💾 Save session", type="primary", disabled=not can_save):
        s_name = session_name.strip()
        v_name = version_name.strip()
        anonymized_cv = _anonymize_cv(cv)
        anonymized_application = _anonymize_application(application)
        anonymized_job = _anonymize_application(job_posting)
        for filename, content in zip(FILENAMES, (anonymized_cv, anonymized_application, anonymized_job)):
            storage.save(s_name, v_name, filename, content)
        st.session_state["saved_cv"] = anonymized_cv
        st.session_state["saved_application"] = anonymized_application
        st.session_state["saved_job"] = anonymized_job
        cv_fields = len(re.findall(r"\\def\\[a-zA-Z]+\{REDACTED\}", anonymized_cv))
        app_fields = len(re.findall(r"REDACTED", anonymized_application))
        st.success(f"Saved to {s_name}/{v_name}/ — {cv_fields} CV field(s) and {app_fields} contact detail(s) anonymised")
    elif not can_save:
        st.caption("Enter a session name and version name to enable saving.")

with tab_viewer:
    st.markdown(
        "<style>[data-testid='stCode'] pre { white-space: pre-wrap; word-break: break-word; }</style>",
        unsafe_allow_html=True,
    )
    saved_cv = st.session_state.get("saved_cv", "")
    saved_application = st.session_state.get("saved_application", "")
    saved_job = st.session_state.get("saved_job", "")

    view_mode = st.radio("Mode", ["Single version", "Compare two versions"], horizontal=True, label_visibility="collapsed")

    if view_mode == "Single version":
        if not any([saved_cv, saved_job, saved_application]):
            st.info("Select a saved session/version above to view its files.")
        else:
            sub_cv, sub_job, sub_app = st.tabs(["CV", "Job Posting", "Application Letter"])
            with sub_cv:
                st.code(saved_cv, language="latex", line_numbers=True) if saved_cv else st.caption("No CV saved.")
            with sub_job:
                if saved_job:
                    _backends = []
                    if MODEL_DIR.exists():
                        _backends.append(_BACKEND_SPACY)
                    if _llm_available():
                        _backends.append(_BACKEND_LLM)

                    ctrl_cols = st.columns(3)
                    with ctrl_cols[0]:
                        job_view = st.toggle("Rendered", value=True, key="job_rendered")
                    with ctrl_cols[1]:
                        skill_hl = st.toggle(
                            "Highlight skills",
                            value=bool(_backends),
                            disabled=not _backends,
                            key="skill_highlight",
                        )
                    with ctrl_cols[2]:
                        backend = st.selectbox(
                            "Backend",
                            _backends or ["—"],
                            disabled=not _backends or not skill_hl,
                            key="skill_backend",
                            label_visibility="collapsed",
                        )
                    if job_view:
                        if skill_hl and _backends:
                            entities = _BACKEND_DISPATCH[backend](saved_job)
                            html = highlight_html(saved_job, entities)
                            legend = " ".join(
                                f'<span style="background:{c};padding:1px 6px;border-radius:3px;margin-right:6px;">{lbl}</span>'
                                for lbl, c in LABEL_COLOURS.items()
                            )
                            st.markdown(legend, unsafe_allow_html=True)
                            st.markdown(html, unsafe_allow_html=True)
                            counts = Counter((e["label"], e["text"]) for e in entities)
                            with st.expander(f"Detected skills ({len(entities)})"):
                                for (label, text), n in counts.most_common():
                                    st.write(f"**{label}** — {text}" + (f" ×{n}" if n > 1 else ""))
                        else:
                            st.markdown(saved_job)
                    else:
                        st.code(saved_job, language="latex", line_numbers=True)
                else:
                    st.caption("No job posting saved.")
            with sub_app:
                st.code(saved_application, language="latex", line_numbers=True) if saved_application else st.caption("No application letter saved.")
    else:
        cmp_cols = st.columns(2)
        # Default A to current session/version if loaded
        default_a = 0
        if selected_session != _NEW_SESSION and selected_version != _NEW_VERSION:
            try:
                default_a = _ALL_SV.index((selected_session, selected_version)) + 1
            except ValueError:
                pass
        with cmp_cols[0]:
            cmp_a_idx = st.selectbox("Version A", range(-1, len(_ALL_SV)),
                                     format_func=lambda i: "— pick —" if i < 0 else _SV_LABELS[i],
                                     index=default_a, key="cmp_a")
        with cmp_cols[1]:
            cmp_b_idx = st.selectbox("Version B", range(-1, len(_ALL_SV)),
                                     format_func=lambda i: "— pick —" if i < 0 else _SV_LABELS[i],
                                     key="cmp_b")

        if cmp_a_idx >= 0 and cmp_b_idx >= 0:
            sv_a = _ALL_SV[cmp_a_idx]
            sv_b = _ALL_SV[cmp_b_idx]
            for fname in FILENAMES:
                text_a = storage.load(*sv_a, fname)
                text_b = storage.load(*sv_b, fname)
                label_a = _session_version_label(*sv_a)
                label_b = _session_version_label(*sv_b)
                diff_lines = list(difflib.unified_diff(
                    text_a.splitlines(keepends=True),
                    text_b.splitlines(keepends=True),
                    fromfile=f"{label_a}/{fname}",
                    tofile=f"{label_b}/{fname}",
                ))
                n_changed = sum(1 for l in diff_lines if l.startswith(("+ ", "- ")))
                label = f"{fname} — {'no differences' if not diff_lines else f'{n_changed} changed lines'}"
                with st.expander(label, expanded=bool(diff_lines)):
                    if diff_lines:
                        st.code("".join(diff_lines), language="diff")
                    else:
                        st.caption("Files are identical.")

with tab_shell:
    mirror = _mirror_dir()

    if "claude_history" not in st.session_state:
        st.session_state.claude_history = []

    if st.button("↺ Restart shell", help="Reload Claude with the current list of sessions"):
        st.session_state.claude_history = []
        st.rerun()

    # --- Input / Output pickers (session/version) ---
    io_cols = st.columns(2)
    with io_cols[0]:
        st.caption("Input (context for Claude)")
        in_idx = st.selectbox("Input version", range(-1, len(_ALL_SV)),
                              format_func=lambda i: "— none —" if i < 0 else _SV_LABELS[i],
                              key="in_sv", label_visibility="collapsed")
        in_file = st.selectbox("Input file", list(_FILE_LABELS.keys()), format_func=lambda k: _FILE_LABELS[k], key="in_file", label_visibility="collapsed")

    with io_cols[1]:
        st.caption("Output (save Claude's last reply)")
        out_options = [_NEW_VERSION] + _SV_LABELS
        out_choice = st.selectbox("Output version", out_options, key="out_sv", label_visibility="collapsed")
        out_file = st.selectbox("Output file", list(_FILE_LABELS.keys()), format_func=lambda k: _FILE_LABELS[k], key="out_file", label_visibility="collapsed")
        out_new_session = ""
        out_new_version = ""
        if out_choice == _NEW_VERSION:
            out_new_cols = st.columns(2)
            with out_new_cols[0]:
                out_new_session = st.text_input("Session", placeholder="e.g. company-role", key="out_new_sess", label_visibility="collapsed")
            with out_new_cols[1]:
                out_new_version = st.text_input("Version", placeholder="e.g. final", key="out_new_ver", label_visibility="collapsed")

    st.divider()

    if not st.session_state.claude_history:
        with st.spinner("Starting Claude…"):
            greeting = _run_claude(
                storage, mirror,
                "Introduce yourself briefly and list the available session folders and their files."
            )
        st.session_state.claude_history.append(("assistant", greeting))

    selected_prompt = st.selectbox(
        "Quick actions",
        options=list(_QUICK_PROMPTS.keys()),
        format_func=lambda k: _QUICK_PROMPTS[k] if k == "" else k,
        label_visibility="collapsed",
    )

    for role, msg in st.session_state.claude_history:
        st.chat_message(role).markdown(msg)

    user_input = st.chat_input("Ask Claude…")
    if selected_prompt and selected_prompt != st.session_state.get("_last_quick_prompt"):
        st.session_state["_last_quick_prompt"] = selected_prompt
        user_input = _QUICK_PROMPTS[selected_prompt]

    if user_input:
        st.session_state.claude_history.append(("user", user_input))
        st.chat_message("user").markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                recent = st.session_state.claude_history[-(_MAX_HISTORY_TURNS * 2 + 1):-1]
                history_ctx = "\n".join(
                    f"{'User' if r == 'user' else 'Assistant'}: {m}" for r, m in recent
                )
                # Prepend input file content if selected
                input_ctx = ""
                if in_idx >= 0:
                    in_s, in_v = _ALL_SV[in_idx]
                    input_content = storage.load(in_s, in_v, in_file)
                    if input_content:
                        input_ctx = f"Context from {in_s}/{in_v}/{in_file}:\n```\n{input_content}\n```\n\n"
                full_prompt = f"{history_ctx}\nUser: {input_ctx}{user_input}" if history_ctx else f"{input_ctx}{user_input}"
                # Resolve output target
                out_target: tuple[str, str, str] | None = None
                if out_choice != _NEW_VERSION:
                    out_sv_idx = _SV_LABELS.index(out_choice)
                    out_s, out_v = _ALL_SV[out_sv_idx]
                    out_target = (out_s, out_v, out_file)
                elif out_new_session.strip() and out_new_version.strip():
                    out_target = (out_new_session.strip(), out_new_version.strip(), out_file)
                reply = _run_claude(storage, mirror, full_prompt, output_target=out_target)
            st.markdown(reply)
        st.session_state.claude_history.append(("assistant", reply))

