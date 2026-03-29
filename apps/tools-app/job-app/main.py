import difflib
import re
import shutil
import subprocess
from pathlib import Path
import streamlit as st

_HERE = Path(__file__).parent
_DATA = _HERE / "data"
_DATA.mkdir(exist_ok=True)

_PERSONAL_FIELDS = {"name", "phone", "city", "email", "address", "linkedin", "github", "website", "mobile"}
_CLAUDE = shutil.which("claude") or "/opt/homebrew/bin/claude"
_MAX_HISTORY_TURNS = 10
_NEW_SESSION = "— new session —"


def _load(session: str, filename: str) -> str:
    path = _DATA / session / filename
    return path.read_text(encoding="utf-8") if path.exists() else ""


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


def _run_claude(prompt: str, output_path: Path | None = None) -> str:
    write_instruction = (
        f"When writing files, ONLY write to {output_path}. "
        "Never modify the files you read from — treat them as read-only source material."
        if output_path else
        "Do not write or modify any files unless explicitly asked."
    )
    system = (
        "You are a job application assistant. "
        f"The working directory is {_DATA}, which contains session subfolders "
        "each with cv.tex, application.tex, and job_posting.tex. "
        "Help the user craft, review, and improve their job applications. "
        + write_instruction
    )
    result = subprocess.run(
        [_CLAUDE, "-p", prompt, "--system-prompt", system, "--add-dir", str(_DATA),
         "--allowedTools", "Edit", "Write", "Read", "Bash"],
        cwd=_DATA,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return (result.stdout or result.stderr).strip()


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

# --- Page ---

st.title("Job Application")

sessions = sorted([d.name for d in _DATA.iterdir() if d.is_dir()])

col_load, col_name = st.columns([2, 3])
with col_load:
    selected_session = st.selectbox(
        "Load session",
        options=[_NEW_SESSION] + sessions,
        index=0,
    )
with col_name:
    default_name = "" if selected_session == _NEW_SESSION else selected_session
    session_name = st.text_input("Session name", value=default_name, placeholder="e.g. company-role-2024")

# Load files once when selected session changes; cache for both editor and viewer
if st.session_state.get("_last_session") != selected_session:
    st.session_state["_last_session"] = selected_session
    if selected_session != _NEW_SESSION:
        st.session_state["cv_text"] = _load(selected_session, "cv.tex")
        st.session_state["application_text"] = _load(selected_session, "application.tex")
        st.session_state["job_text"] = _load(selected_session, "job_posting.tex")
        st.session_state["saved_cv"] = st.session_state["cv_text"]
        st.session_state["saved_application"] = st.session_state["application_text"]
        st.session_state["saved_job"] = st.session_state["job_text"]
    else:
        for key in ("cv_text", "application_text", "job_text", "saved_cv", "saved_application", "saved_job"):
            st.session_state[key] = ""

st.divider()

tab_editor, tab_viewer, tab_shell = st.tabs(["✏️ Editor", "🔍 Viewer", "🖥️ Shell"])

with tab_editor:
    if selected_session == _NEW_SESSION and sessions:
        with st.expander("Copy content from existing session"):
            copy_cols = st.columns([2, 1, 1])
            with copy_cols[0]:
                copy_source = st.selectbox("Source session", sessions, label_visibility="collapsed")
            with copy_cols[1]:
                copy_cv = st.checkbox("CV", value=True)
            with copy_cols[2]:
                copy_app = st.checkbox("Application Letter")
            if st.button("Copy"):
                if copy_cv:
                    st.session_state["cv_text"] = _load(copy_source, "cv.tex")
                if copy_app:
                    st.session_state["application_text"] = _load(copy_source, "application.tex")
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

    if st.button("💾 Save session", type="primary", disabled=not session_name.strip()):
        name = session_name.strip()
        session_dir = _DATA / name
        session_dir.mkdir(parents=True, exist_ok=True)
        anonymized_cv = _anonymize_cv(cv)
        anonymized_application = _anonymize_application(application)
        anonymized_job = _anonymize_application(job_posting)
        (session_dir / "cv.tex").write_text(anonymized_cv, encoding="utf-8")
        (session_dir / "application.tex").write_text(anonymized_application, encoding="utf-8")
        (session_dir / "job_posting.tex").write_text(anonymized_job, encoding="utf-8")
        st.session_state["saved_cv"] = anonymized_cv
        st.session_state["saved_application"] = anonymized_application
        st.session_state["saved_job"] = anonymized_job
        cv_fields = len(re.findall(r"\\def\\[a-zA-Z]+\{REDACTED\}", anonymized_cv))
        app_fields = len(re.findall(r"REDACTED", anonymized_application))
        st.success(f"Saved to data/{name}/ — {cv_fields} CV field(s) and {app_fields} contact detail(s) anonymised")
    elif not session_name.strip():
        st.caption("Enter a session name to enable saving.")

with tab_viewer:
    st.markdown(
        "<style>[data-testid='stCode'] pre { white-space: pre-wrap; word-break: break-word; }</style>",
        unsafe_allow_html=True,
    )
    saved_cv = st.session_state.get("saved_cv", "")
    saved_application = st.session_state.get("saved_application", "")
    saved_job = st.session_state.get("saved_job", "")

    view_mode = st.radio("Mode", ["Single session", "Compare two sessions"], horizontal=True, label_visibility="collapsed")

    if view_mode == "Single session":
        if not any([saved_cv, saved_job, saved_application]):
            st.info("Select a saved session above to view its files.")
        else:
            sub_cv, sub_job, sub_app = st.tabs(["CV", "Job Posting", "Application Letter"])
            with sub_cv:
                st.code(saved_cv, language="latex", line_numbers=True) if saved_cv else st.caption("No CV saved.")
            with sub_job:
                st.code(saved_job, language="latex", line_numbers=True) if saved_job else st.caption("No job posting saved.")
            with sub_app:
                st.code(saved_application, language="latex", line_numbers=True) if saved_application else st.caption("No application letter saved.")
    else:
        all_files = {s: list(((_DATA / s)).iterdir()) for s in sessions}
        cmp_cols = st.columns(2)
        default_a = sessions.index(selected_session) + 1 if selected_session != _NEW_SESSION and selected_session in sessions else 0
        with cmp_cols[0]:
            cmp_a = st.selectbox("Session A", ["— pick —"] + sessions, index=default_a, key="cmp_a")
        with cmp_cols[1]:
            cmp_b = st.selectbox("Session B", ["— pick —"] + sessions, key="cmp_b")

        if cmp_a != "— pick —" and cmp_b != "— pick —":
            all_filenames = sorted({f.name for s in [cmp_a, cmp_b] for f in (_DATA / s).iterdir() if f.is_file()})
            for fname in all_filenames:
                path_a = _DATA / cmp_a / fname
                path_b = _DATA / cmp_b / fname
                text_a = path_a.read_text(encoding="utf-8") if path_a.exists() else ""
                text_b = path_b.read_text(encoding="utf-8") if path_b.exists() else ""
                diff_lines = list(difflib.unified_diff(
                    text_a.splitlines(keepends=True),
                    text_b.splitlines(keepends=True),
                    fromfile=f"{cmp_a}/{fname}",
                    tofile=f"{cmp_b}/{fname}",
                ))
                n_changed = sum(1 for l in diff_lines if l.startswith(("+ ", "- ")))
                label = f"{fname} — {'no differences' if not diff_lines else f'{n_changed} changed lines'}"
                with st.expander(label, expanded=bool(diff_lines)):
                    if diff_lines:
                        st.code("".join(diff_lines), language="diff")
                    else:
                        st.caption("Files are identical.")

_FILE_LABELS = {"cv.tex": "CV", "application.tex": "Application Letter", "job_posting.tex": "Job Posting"}
_OUTPUT_FILE_LABELS = {
    "cv_response.tex": "CV (response)",
    "application_response.tex": "Application Letter (response)",
    "cv.tex": "CV (overwrite)",
    "application.tex": "Application Letter (overwrite)",
    "job_posting.tex": "Job Posting (overwrite)",
}

with tab_shell:
    if "claude_history" not in st.session_state:
        st.session_state.claude_history = []

    if st.button("↺ Restart shell", help="Reload Claude with the current list of sessions"):
        st.session_state.claude_history = []
        st.rerun()

    # --- Input / Output file pickers ---
    io_cols = st.columns(2)
    with io_cols[0]:
        st.caption("Input file (context for Claude)")
        in_sessions = ["— none —"] + sessions
        in_session = st.selectbox("Input session", in_sessions, key="in_session", label_visibility="collapsed")
        in_file = st.selectbox("Input file", list(_FILE_LABELS.keys()), format_func=_FILE_LABELS.get, key="in_file", label_visibility="collapsed")

    with io_cols[1]:
        st.caption("Output file (save Claude's last reply)")
        out_sessions = ["— new session —"] + sessions
        out_session = st.selectbox("Output session", out_sessions, key="out_session", label_visibility="collapsed")
        out_file = st.selectbox("Output file", list(_OUTPUT_FILE_LABELS.keys()), format_func=_OUTPUT_FILE_LABELS.get, key="out_file", label_visibility="collapsed")
        out_name = ""
        if out_session == "— new session —":
            out_name = st.text_input("New session name", placeholder="e.g. company-role-2024", key="out_name", label_visibility="collapsed")

    st.divider()

    if not st.session_state.claude_history:
        with st.spinner("Starting Claude…"):
            greeting = _run_claude(
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
                if in_session != "— none —":
                    input_path = _DATA / in_session / in_file
                    if input_path.exists():
                        input_ctx = f"Context from {in_session}/{in_file}:\n```\n{input_path.read_text(encoding='utf-8')}\n```\n\n"
                full_prompt = f"{history_ctx}\nUser: {input_ctx}{user_input}" if history_ctx else f"{input_ctx}{user_input}"
                out_target = None
                if out_session != "— new session —" and out_session:
                    out_target = _DATA / out_session / out_file
                elif out_name.strip():
                    out_target = _DATA / out_name.strip() / out_file
                reply = _run_claude(full_prompt, output_path=out_target)
            st.markdown(reply)
        st.session_state.claude_history.append(("assistant", reply))

    # --- Save last reply to output file ---
    last_reply = next((m for r, m in reversed(st.session_state.claude_history) if r == "assistant"), None)
    if last_reply:
        target_session = out_name.strip() if out_session == "— new session —" else out_session
        save_disabled = not target_session
        if st.button(f"💾 Save reply → {target_session or '…'}/{_OUTPUT_FILE_LABELS[out_file]}", disabled=save_disabled):
            out_dir = _DATA / target_session
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / out_file
            if out_file in ("cv_response.tex", "application_response.tex"):
                with out_path.open("a", encoding="utf-8") as f:
                    if out_path.stat().st_size > 0 if out_path.exists() else False:
                        f.write("\n\n% ---\n\n")
                    f.write(last_reply)
            else:
                out_path.write_text(last_reply, encoding="utf-8")
            st.success(f"Saved to data/{target_session}/{out_file}")

    # --- Diff: input file vs output file ---
    if in_session != "— none —":
        input_path = _DATA / in_session / in_file
        out_target_session = out_name.strip() if out_session == "— new session —" else out_session
        output_path = _DATA / out_target_session / out_file if out_target_session else None
        if input_path.exists() and output_path and output_path.exists():
            st.divider()
            st.subheader("Diff — input vs output")
            diff_lines = list(difflib.unified_diff(
                input_path.read_text(encoding="utf-8").splitlines(keepends=True),
                output_path.read_text(encoding="utf-8").splitlines(keepends=True),
                fromfile=f"{in_session}/{in_file}",
                tofile=f"{out_target_session}/{out_file}",
            ))
            if diff_lines:
                st.code("".join(diff_lines), language="diff")
            else:
                st.caption("No differences.")
