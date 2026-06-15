/**
 * App.jsx — the entire single-page UI for the Muse practice tool.
 *
 * Two-panel layout: a live chat with the AI "mentee" (left) and a private
 * coaching "whisper" feed from Muse (right). Unauthenticated users see a
 * Sign In / Create Account screen first; once logged in, the chat speaks the
 * backend /ws JSON protocol directly:
 *   incoming  - {type:"history"|"token"|"done"|"whisper", ...}
 *   outgoing  - raw text (the mentor's message)
 *
 * Responsibilities held here: auth gate (localStorage + /auth/me), WebSocket
 * lifecycle with token, streaming-token assembly, optional voice I/O, and
 * autoscroll. No router — state is local because the app is one screen.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import { COUNTRIES } from './countries';

const API_BASE = 'http://localhost:8000';

async function parseApiError(res, fallback) {
  try {
    const data = await res.json();
    if (typeof data.detail === 'string') return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail.map((d) => d.msg || JSON.stringify(d)).join(', ');
    }
    return fallback;
  } catch {
    return res.ok ? fallback : `Server error (${res.status}). Please try again.`;
  }
}

// Display metadata per message role. `assistant` is the AI mentee ("Alex");
// `user` is the human mentor. Mentor name is overridden with user.first_name.
const SPEAKERS = {
  user: { name: 'You', sub: 'Mentor', initial: 'Y' },
  assistant: { name: 'Alex', sub: 'Mentee', initial: 'A' },
};

function renderWhisper(text) {
  const parts = text.split(/(".*?")/g);
  return parts.map((part, i) =>
    part.length >= 2 && part.startsWith('"') && part.endsWith('"')
      ? <em key={i}>{part}</em>
      : <span key={i}>{part}</span>
  );
}

function speak(text) {
  if (!text || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1.0;
  utter.pitch = 1.05;
  window.speechSynthesis.speak(utter);
}

export default function App() {
  // ---- Auth state ---------------------------------------------------------
  const [user, setUser] = useState(null);
  const [accessToken, setAccessToken] = useState(null);
  const [authTab, setAuthTab] = useState('signin');
  const [authError, setAuthError] = useState(null);
  const [authLoading, setAuthLoading] = useState(false);
  const [booting, setBooting] = useState(true);

  // Sign-in form
  const [signInEmail, setSignInEmail] = useState('');
  const [signInPassword, setSignInPassword] = useState('');

  // Register form
  const [regFirstName, setRegFirstName] = useState('');
  const [regLastName, setRegLastName] = useState('');
  const [regEmail, setRegEmail] = useState('');
  const [regPassword, setRegPassword] = useState('');
  const [regDob, setRegDob] = useState('');
  const [regLocation, setRegLocation] = useState('');
  const [regNationality, setRegNationality] = useState('');

  // ---- Chat state ---------------------------------------------------------
  const [messages, setMessages] = useState([]);
  const [whispers, setWhispers] = useState([]);
  const [reflecting, setReflecting] = useState(false);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [voiceOn, setVoiceOn] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);

  // ---- Delete-account state -----------------------------------------------
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deletePassword, setDeletePassword] = useState('');
  const [deleteError, setDeleteError] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const wsRef = useRef(null);
  const replyRef = useRef('');
  const voiceOnRef = useRef(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const messagesEndRef = useRef(null);
  const whispersEndRef = useRef(null);

  const clearAuth = useCallback(() => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    setUser(null);
    setAccessToken(null);
    setMessages([]);
    setWhispers([]);
  }, []);

  const handleExpired = useCallback(() => {
    clearAuth();
  }, [clearAuth]);

  useEffect(() => { voiceOnRef.current = voiceOn; }, [voiceOn]);

  // ---- Boot: OAuth query capture + session restore via /auth/me ------------
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const urlAccess = params.get('access_token');
    const urlRefresh = params.get('refresh_token');
    if (urlAccess && urlRefresh) {
      localStorage.setItem('access_token', urlAccess);
      localStorage.setItem('refresh_token', urlRefresh);
      window.history.replaceState({}, '', '/');
    }

    const token = localStorage.getItem('access_token');
    if (!token) {
      setBooting(false);
      return;
    }

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/auth/me`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) throw new Error('invalid');
        const profile = await res.json();
        setUser({
          id: profile.id,
          email: profile.email,
          first_name: profile.first_name,
          last_name: profile.last_name,
        });
        setAccessToken(token);
      } catch {
        clearAuth();
      } finally {
        setBooting(false);
      }
    })();
  }, [clearAuth]);

  // ---- WebSocket (only when authenticated) --------------------------------
  useEffect(() => {
    if (!accessToken) return;

    const ws = new WebSocket(`${API_BASE.replace('http', 'ws')}/ws?token=${accessToken}`);
    ws.onerror = () => { setLoading(false); setReflecting(false); };
    ws.onclose = (event) => {
      setLoading(false);
      setReflecting(false);
      if (event.code === 4001) handleExpired();
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'history') {
        setMessages(data.messages);
        if (data.whispers) {
          setWhispers(data.whispers.map((w) => ({ label: w.label || 'Insight', content: w.content })));
        }
      } else if (data.type === 'token') {
        replyRef.current += data.content;
        setMessages((prev) => {
          const updated = [...prev];
          if (updated.length > 0 && updated[updated.length - 1].role === 'assistant') {
            updated[updated.length - 1].content += data.content;
          }
          return updated;
        });
      } else if (data.type === 'done') {
        setLoading(false);
        setReflecting(true);
        if (voiceOnRef.current) speak(replyRef.current);
        replyRef.current = '';
      } else if (data.type === 'whisper') {
        setReflecting(false);
        setWhispers((prev) => [...prev, { label: data.label || 'Insight', content: data.content }]);
      }
    };

    wsRef.current = ws;
    return () => ws.close();
  }, [accessToken, handleExpired]);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);
  useEffect(() => { whispersEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [whispers, reflecting]);

  // ---- Auth handlers ------------------------------------------------------
  const persistSession = (data) => {
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('refresh_token', data.refresh_token);
    setUser(data.user);
    setAccessToken(data.access_token);
    setAuthError(null);
  };

  const signIn = async (e) => {
    e.preventDefault();
    setAuthError(null);
    setAuthLoading(true);
    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: signInEmail, password: signInPassword }),
      });
      if (!res.ok) throw new Error(await parseApiError(res, 'Sign in failed'));
      const data = await res.json();
      persistSession(data);
    } catch (err) {
      setAuthError(err.message || 'Sign in failed');
    } finally {
      setAuthLoading(false);
    }
  };

  const register = async (e) => {
    e.preventDefault();
    setAuthError(null);
    setAuthLoading(true);
    try {
      const res = await fetch(`${API_BASE}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: regEmail,
          password: regPassword,
          first_name: regFirstName,
          last_name: regLastName,
          dob: regDob,
          location: regLocation.trim() || null,
          nationality: regNationality || null,
        }),
      });
      if (!res.ok) throw new Error(await parseApiError(res, 'Registration failed'));
      const data = await res.json();
      persistSession(data);
    } catch (err) {
      setAuthError(err.message || 'Registration failed');
    } finally {
      setAuthLoading(false);
    }
  };

  const logout = async () => {
    const refreshToken = localStorage.getItem('refresh_token');
    if (refreshToken) {
      try {
        await fetch(`${API_BASE}/auth/logout`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
      } catch {
        // Best-effort revoke; still clear local session.
      }
    }
    clearAuth();
  };

  const googleLogin = () => {
    window.location.href = `${API_BASE}/auth/google`;
  };

  const openDeleteModal = () => {
    setDeletePassword('');
    setDeleteError(null);
    setShowDeleteModal(true);
  };

  const closeDeleteModal = () => {
    if (deleting) return;
    setShowDeleteModal(false);
    setDeletePassword('');
    setDeleteError(null);
  };

  const deleteAccount = async () => {
    setDeleteError(null);
    setDeleting(true);
    try {
      const token = localStorage.getItem('access_token');
      const res = await fetch(`${API_BASE}/auth/me`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ password: deletePassword || null }),
      });
      if (!res.ok) throw new Error(await parseApiError(res, 'Account deletion failed'));
      // Tear down the live socket before clearing the session so it doesn't try
      // to reconnect with a now-deleted account.
      wsRef.current?.close();
      setShowDeleteModal(false);
      clearAuth();
    } catch (err) {
      setDeleteError(err.message || 'Account deletion failed');
    } finally {
      setDeleting(false);
    }
  };

  // ---- Chat handlers ------------------------------------------------------
  const sendMessage = (text) => {
    const trimmed = text.trim();
    if (!trimmed || loading) return;
    setReflecting(false);
    replyRef.current = '';
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: trimmed },
      { role: 'assistant', content: '' },
    ]);
    setLoading(true);
    wsRef.current?.send(trimmed);
  };

  const handleSend = () => {
    if (!input.trim()) return;
    sendMessage(input);
    setInput('');
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await transcribeAndSend(blob);
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setRecording(true);
    } catch (err) {
      console.error('Mic error:', err);
      alert('Could not access the microphone.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const transcribeAndSend = async (blob) => {
    setTranscribing(true);
    const form = new FormData();
    form.append('audio', blob, 'audio.webm');
    try {
      const res = await fetch(`${API_BASE}/transcribe`, { method: 'POST', body: form });
      const data = await res.json();
      const text = (data.text || '').trim();
      if (text) sendMessage(text);
    } catch (err) {
      console.error('Transcription request failed:', err);
    } finally {
      setTranscribing(false);
    }
  };

  // ---- Render gate --------------------------------------------------------
  if (booting) return null;

  if (!user) {
    return (
      <div className="auth-screen">
        <div className="auth-card">
          <div className="auth-tabs">
            <button
              type="button"
              className={`auth-tab ${authTab === 'signin' ? 'active' : ''}`}
              onClick={() => { setAuthTab('signin'); setAuthError(null); }}
              disabled={authLoading}
            >
              Sign In
            </button>
            <button
              type="button"
              className={`auth-tab ${authTab === 'register' ? 'active' : ''}`}
              onClick={() => { setAuthTab('register'); setAuthError(null); }}
              disabled={authLoading}
            >
              Create Account
            </button>
          </div>

          <button type="button" className="google-btn" onClick={googleLogin} disabled={authLoading}>
            Sign in with Google
          </button>

          <div className="auth-divider"><span>or continue with email</span></div>

          {authTab === 'signin' ? (
            <form className="auth-form" onSubmit={signIn}>
              <input
                type="email"
                placeholder="Email"
                value={signInEmail}
                onChange={(e) => setSignInEmail(e.target.value)}
                disabled={authLoading}
                required
              />
              <input
                type="password"
                placeholder="Password"
                value={signInPassword}
                onChange={(e) => setSignInPassword(e.target.value)}
                disabled={authLoading}
                required
              />
              <button type="submit" className="auth-submit" disabled={authLoading}>
                {authLoading ? 'Signing in…' : 'Sign In'}
              </button>
              <p className="auth-switch">
                Don&apos;t have an account?{' '}
                <button type="button" onClick={() => { setAuthTab('register'); setAuthError(null); }} disabled={authLoading}>
                  Create one
                </button>
              </p>
            </form>
          ) : (
            <form className="auth-form" onSubmit={register}>
              <div className="name-row">
                <input
                  type="text"
                  placeholder="First name"
                  value={regFirstName}
                  onChange={(e) => setRegFirstName(e.target.value)}
                  disabled={authLoading}
                  required
                />
                <input
                  type="text"
                  placeholder="Last name"
                  value={regLastName}
                  onChange={(e) => setRegLastName(e.target.value)}
                  disabled={authLoading}
                  required
                />
              </div>
              <input
                type="email"
                placeholder="Email"
                value={regEmail}
                onChange={(e) => setRegEmail(e.target.value)}
                disabled={authLoading}
                required
              />
              <input
                type="password"
                placeholder="Password"
                value={regPassword}
                onChange={(e) => setRegPassword(e.target.value)}
                disabled={authLoading}
                required
              />
              <input
                type="date"
                value={regDob}
                onChange={(e) => setRegDob(e.target.value)}
                disabled={authLoading}
                required
              />
              <input
                type="text"
                placeholder="Location (optional)"
                value={regLocation}
                onChange={(e) => setRegLocation(e.target.value)}
                disabled={authLoading}
              />
              <select
                value={regNationality}
                onChange={(e) => setRegNationality(e.target.value)}
                disabled={authLoading}
                className="auth-select"
              >
                <option value="">Nationality (optional)</option>
                {COUNTRIES.map((country) => (
                  <option key={country} value={country}>{country}</option>
                ))}
              </select>
              <button type="submit" className="auth-submit" disabled={authLoading}>
                {authLoading ? 'Creating account…' : 'Create Account'}
              </button>
              <p className="auth-switch">
                Already have an account?{' '}
                <button type="button" onClick={() => { setAuthTab('signin'); setAuthError(null); }} disabled={authLoading}>
                  Sign in
                </button>
              </p>
            </form>
          )}

          {authError && <p className="auth-error">{authError}</p>}
        </div>
      </div>
    );
  }

  const mentorInitial = user.first_name?.charAt(0)?.toUpperCase() || 'Y';

  return (
    <div className="app">
      <div className="conversation-panel">
        <div className="chat-header">
          <div className="contact">
            <div className="avatar lg">A</div>
            <div className="contact-meta">
              <div className="contact-name">Alex</div>
              <div className="contact-role">Mentee · practice partner</div>
            </div>
          </div>
          <div className="header-actions">
            <button
              className="voice-toggle"
              onClick={() => { if (voiceOn) window.speechSynthesis.cancel(); setVoiceOn((v) => !v); }}
            >
              {voiceOn ? '🔊 Voice on' : '🔈 Voice off'}
            </button>
            <button type="button" className="logout-btn" onClick={logout}>
              Log out
            </button>
            <button type="button" className="delete-account-btn" onClick={openDeleteModal}>
              Delete account
            </button>
          </div>
        </div>

        <div className="messages">
          {messages.map((msg, idx) => {
            const sp = SPEAKERS[msg.role] || SPEAKERS.assistant;
            const name = msg.role === 'user' ? user.first_name : sp.name;
            const initial = msg.role === 'user' ? mentorInitial : sp.initial;
            return (
              <div key={idx} className={`message ${msg.role}`}>
                <div className="avatar">{initial}</div>
                <div className="bubble-col">
                  <div className="speaker-name">{name}<span className="speaker-sub"> · {sp.sub}</span></div>
                  <div className="message-content">{msg.content}</div>
                </div>
              </div>
            );
          })}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <button
            className={`mic-btn ${recording ? 'recording' : ''}`}
            onClick={recording ? stopRecording : startRecording}
            disabled={loading || transcribing}
            title="Speak as the mentor"
          >
            {recording ? '■' : '🎤'}
          </button>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder={transcribing ? 'Transcribing…' : recording ? 'Listening…' : 'Respond as the mentor…'}
            disabled={loading || recording || transcribing}
          />
          <button onClick={handleSend} disabled={loading || recording || transcribing}>Send</button>
        </div>
      </div>

      <div className="whisper-panel">
        <div className="muse-header">
          <div className="muse-title">✦ Muse</div>
          <div className="muse-sub">Private to you · the mentee can't see this</div>
        </div>
        <div className="whispers">
          {whispers.length === 0 && !reflecting && (
            <div className="whisper-empty">Muse is listening. Coaching appears here after each exchange.</div>
          )}
          {whispers.map((w, idx) => (
            <div key={idx} className="whisper-card">
              <div className="whisper-tag">{w.label}</div>
              <div className="whisper-body">{renderWhisper(w.content)}</div>
            </div>
          ))}
          {reflecting && (
            <div className="reflecting"><span className="muse-mark">✦</span> Muse is reflecting<span>…</span></div>
          )}
          <div ref={whispersEndRef} />
        </div>
      </div>

      {showDeleteModal && (
        <div className="modal-overlay" onClick={closeDeleteModal}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-title">Delete your account?</div>
            <p className="modal-body">
              This permanently erases your account and all of your data — chat history,
              private Muse coaching notes, and long-term memory. This cannot be undone.
            </p>
            <input
              type="password"
              className="modal-input"
              placeholder="Confirm your password"
              value={deletePassword}
              onChange={(e) => setDeletePassword(e.target.value)}
              disabled={deleting}
            />
            {deleteError && <p className="modal-error">{deleteError}</p>}
            <div className="modal-actions">
              <button type="button" className="modal-cancel" onClick={closeDeleteModal} disabled={deleting}>
                Cancel
              </button>
              <button type="button" className="modal-confirm" onClick={deleteAccount} disabled={deleting}>
                {deleting ? 'Deleting…' : 'Delete everything'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
