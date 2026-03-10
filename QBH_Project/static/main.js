// ─── Module-Level Feedback Session State ─────────────────────────────────────
let _qbhSession = {
    queryId: null,
    internalNames: [],   // exact shown_list in display order
    qType: '',
    excludedSongs: [],   // accumulates across retries in a session
    retryDepth: 0,
    isRetryMode: false,
};
const MAX_RETRY_DEPTH = 2;

// Standard headers for all API calls (skips ngrok intercept page)
const QBH_HEADERS = {
    'Content-Type': 'application/json',
    'ngrok-skip-browser-warning': '69420'
};

// ─── Global onclick hooks (called from HTML) ──────────────────────────────────
function qbhRetryExcluding() { _doRetryExcluding(); }
function qbhFullReset() { _doFullReset(); }

let _doRetryExcluding = () => { };
let _doFullReset = () => { };

// ─── Toast notification ───────────────────────────────────────────────────────
function qbhShowToast(msg, success = true) {
    let toast = document.getElementById('feedbackToast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'feedbackToast';
        toast.style.cssText = [
            'position:fixed;bottom:28px;left:50%;transform:translateX(-50%)',
            'padding:12px 24px;border-radius:24px;font-size:0.95rem',
            'font-weight:600;z-index:9999;transition:all 0.4s ease',
            'pointer-events:none;white-space:nowrap;box-shadow:0 10px 30px rgba(0,0,0,0.5)'
        ].join(';');
        document.body.appendChild(toast);
    }
    toast.style.background = success ? '#1ed760' : '#ff4757';
    toast.style.color = success ? '#000' : '#fff';
    toast.textContent = msg;
    toast.style.opacity = '1';
    toast.style.transform = 'translateX(-50%) translateY(0)';
    clearTimeout(toast._t);
    toast._t = setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(-50%) translateY(10px)';
    }, 3000);
}

// ─── Feedback submit ──────────────────────────────────────────────────────────
async function qbhSubmitFeedback(internalName, mode, extra = {}) {
    const melodyScore = typeof extra === 'number' ? extra : (extra.melody_score || 0);
    const body = {
        query_id: _qbhSession.queryId,
        shown_list: _qbhSession.internalNames,
        selected_song: internalName,
        selected_rank: extra.selected_rank || -1,
        mode: mode,
        q_type: _qbhSession.qType,
        melody_score: melodyScore,
        top1_score: extra.top1_score || 0,
        excluded_songs: _qbhSession.excludedSongs,
        retry_depth: _qbhSession.retryDepth,
    };

    try {
        const res = await fetch('/submit-feedback', {
            method: 'POST',
            headers: QBH_HEADERS,
            body: JSON.stringify(body)
        });
        const data = await res.json();
        if (data.success) {
            qbhShowToast(`✓ Feedback recorded! (${mode.replace(/_/g, ' ')})`, true);
        } else {
            qbhShowToast('⚠ ' + (data.error || 'Feedback error'), false);
        }
    } catch (e) {
        qbhShowToast('⚠ Server unreachable. Check connection.', false);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {

    // ── DOM Refs ──────────────────────────────────────────────────────
    const recordBtn = document.getElementById('recordBtn');
    const recorderSection = document.getElementById('recorderSection');
    const visualizerContainer = document.getElementById('visualizerContainer');
    const statusText = document.getElementById('statusText');
    const timerText = document.getElementById('timerText');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const resultsSection = document.getElementById('resultsSection');
    const loadingSection = document.getElementById('loadingSection');
    const topMatchesList = document.getElementById('topMatchesList');
    const datasetRecList = document.getElementById('datasetRecList');
    const datasetRecBlock = document.getElementById('datasetRecBlock');
    const spotifyRecList = document.getElementById('spotifyRecList');
    const spotifyRecBlock = document.getElementById('spotifyRecBlock');
    const albumArt = document.getElementById('albumArt');
    const songTitle = document.getElementById('songTitle');
    const songArtist = document.getElementById('songArtist');
    const songAlbum = document.getElementById('songAlbum');
    const songDate = document.getElementById('songDate');
    const songConfidence = document.getElementById('songConfidence');
    const previewPlayer = document.getElementById('previewPlayer');
    const spotifyLink = document.getElementById('spotifyLink');

    // ── Audio state ───────────────────────────────────────────────────
    let audioContext, sourceNode, scriptProcessor, stream;
    let pcmSamples = [];
    let isRecording = false;
    let timerInterval, timeElapsed = 0;
    const TARGET_SR = 22050, MAX_SECONDS = 10, BUFFER_SIZE = 4096;

    // ── WAV encoder ───────────────────────────────────────────────────
    function encodeWAV(samples, sampleRate) {
        const bufLen = samples.length * 2;
        const buffer = new ArrayBuffer(44 + bufLen);
        const view = new DataView(buffer);
        const w = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
        w(0, 'RIFF'); view.setUint32(4, 36 + bufLen, true); w(8, 'WAVE');
        w(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
        view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true);
        view.setUint16(34, 16, true); w(36, 'data'); view.setUint32(40, bufLen, true);
        let off = 44;
        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true); off += 2;
        }
        return new Blob([buffer], { type: 'audio/wav' });
    }

    function downsample(buf, inRate, outRate) {
        if (inRate === outRate) return buf;
        const ratio = inRate / outRate;
        const out = new Float32Array(Math.round(buf.length / ratio));
        for (let i = 0; i < out.length; i++) {
            const start = Math.round(i * ratio), end = Math.round((i + 1) * ratio);
            let sum = 0, cnt = 0;
            for (let j = start; j < end && j < buf.length; j++, cnt++) sum += buf[j];
            out[i] = cnt ? sum / cnt : 0;
        }
        return out;
    }

    // ── Recording ─────────────────────────────────────────────────────
    async function startRecording() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, sampleRate: TARGET_SR }
            });
        } catch { statusText.textContent = "Microphone access denied."; statusText.style.color = "#ff3333"; return; }

        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SR });
        sourceNode = audioContext.createMediaStreamSource(stream);
        scriptProcessor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);
        pcmSamples = [];

        scriptProcessor.onaudioprocess = e => {
            if (!isRecording) return;
            const rs = downsample(e.inputBuffer.getChannelData(0), audioContext.sampleRate, TARGET_SR);
            for (const s of rs) pcmSamples.push(s);
        };

        sourceNode.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);
        isRecording = true;
        visualizerContainer.classList.add('recording');
        statusText.textContent = _qbhSession.isRetryMode
            ? `Retry ${_qbhSession.retryDepth}/${MAX_RETRY_DEPTH}: Hum louder or differently`
            : "Recording… tap to stop";
        recordBtn.innerHTML = '<i class="fa-solid fa-stop"></i>';
        progressContainer.classList.remove('hidden');
        timeElapsed = 0; updateTimer();
        timerInterval = setInterval(() => { timeElapsed += 0.1; updateTimer(); if (timeElapsed >= MAX_SECONDS) stopRecording(); }, 100);
    }

    function stopRecording() {
        if (!isRecording) return;
        isRecording = false; clearInterval(timerInterval);
        if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor.onaudioprocess = null; }
        if (sourceNode) sourceNode.disconnect();
        if (stream) stream.getTracks().forEach(t => t.stop());
        if (audioContext && audioContext.state !== 'closed') audioContext.close();
        const wavBlob = encodeWAV(new Float32Array(pcmSamples), TARGET_SR);
        pcmSamples = [];
        recorderSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        recordBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
        visualizerContainer.classList.remove('recording');
        sendAudio(wavBlob);
    }

    function updateTimer() {
        const r = Math.max(0, MAX_SECONDS - timeElapsed);
        timerText.textContent = `00:${String(Math.floor(r)).padStart(2, '0')}`;
        progressBar.style.width = `${(timeElapsed / MAX_SECONDS) * 100}%`;
    }

    function resetUI() {
        statusText.textContent = "Tap to start humming";
        statusText.style.color = "var(--text-color)";
        timerText.textContent = "00:10";
        progressBar.style.width = "0%";
        progressContainer.classList.add('hidden');
        pcmSamples = []; timeElapsed = 0;
    }

    // ── sendAudio — skips ngrok intercept ──────────────────────────────
    async function sendAudio(blob) {
        const fd = new FormData();
        fd.append('audio', blob, 'recording.wav');

        let endpoint = '/identify-song';
        if (_qbhSession.isRetryMode) {
            fd.append('excluded_songs', JSON.stringify(_qbhSession.excludedSongs));
            fd.append('retry_depth', String(_qbhSession.retryDepth));
            endpoint = '/identify-song-retry';
        }

        try {
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'ngrok-skip-browser-warning': 'true' },
                body: fd
            });

            if (!res.ok) {
                const text = await res.text();
                showError(`Server error (${res.status}): ${text.substring(0, 50)}`);
                loadingSection.classList.add('hidden');
                return;
            }

            const data = await res.json();
            loadingSection.classList.add('hidden');
            if (data.success) displayResults(data);
            else showError(data.error || 'Matching failed.');
        } catch (e) {
            console.error(e);
            loadingSection.classList.add('hidden');
            showError(`Connection failed: ${e.message || 'Check network'}`);
        }
    }

    // ─── Results renderer ───────────────────────────────────────────
    function _updateIdentifiedSongUI(song) {
        if (!song) return;
        if (song.image) { albumArt.src = song.image; albumArt.classList.remove('hidden'); }
        else albumArt.classList.add('hidden');

        songTitle.textContent = song.title || song.song_name || '—';
        songArtist.textContent = song.artist || 'Unknown artist';
        songAlbum.textContent = song.album || '';
        songDate.textContent = song.release_date || '';
        const identifiedConfidence = Number.isFinite(song.confidence_pct) ? song.confidence_pct : (Number.isFinite(song.confidence) ? song.confidence : 0);
        songConfidence.textContent = `${Math.min(100, Math.round(identifiedConfidence))}% Confidence`;

        if (song.preview_url) { previewPlayer.src = song.preview_url; previewPlayer.classList.remove('hidden'); }
        else previewPlayer.classList.add('hidden');

        if (song.spotify_url) { spotifyLink.href = song.spotify_url; spotifyLink.classList.remove('hidden'); }
        else spotifyLink.classList.add('hidden');

        let existingBtn = document.getElementById('identifiedDetailsBtn');
        if (existingBtn) existingBtn.remove();
        if (song.title || song.song_name) {
            const dBtn = document.createElement('button');
            dBtn.id = 'identifiedDetailsBtn';
            dBtn.className = 'details-btn';
            dBtn.innerHTML = '<i class="fa-solid fa-circle-info"></i> Song Details';
            const searchStr = song.title ? (song.title + ' ' + (song.artist || '')) : song.song_name;
            dBtn.addEventListener('click', () => fetchSongDetails(searchStr));
            document.querySelector('.identified-info')?.appendChild(dBtn);
        }
    }

    function displayResults(data) {
        loadingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        _qbhSession.queryId = data.query_id || null;
        _qbhSession.internalNames = data.internal_names || [];
        _qbhSession.qType = data.q_type || '';

        const idSong = data.identified_song || {};
        _updateIdentifiedSongUI(idSong);

        // ── Top matches with SUB-ROW action layout ──
        topMatchesList.innerHTML = '';
        // ── Top matches with SUB-ROW action layout ──
        topMatchesList.innerHTML = '';
        data.top_matches.forEach((m, i) => {
            const internalName = m.song_name || m.internal_name || 'Unknown';
            const songNameStr = internalName.replace(/\.wav$/, '').replace(/_/g, ' ');

            const li = document.createElement('li');
            li.className = 'match-item';
            li.dataset.song = internalName;

            li.innerHTML = `
                <div class="match-row-header" style="cursor: pointer;">
                    <div class="match-song-info">
                        ${m.image
                    ? `<img src="${m.image}" alt="">`
                    : `<i class="fa-solid fa-${i === 0 ? 'trophy' : 'music'}" style="width:38px;text-align:center;opacity:0.4;"></i>`}
                        <div class="match-text">
                            <span class="song-name">${songNameStr}</span>
                            <span class="song-artist">${m.artist || 'Unknown'}</span>
                        </div>
                    </div>
                    <div class="match-stats">
                        <span class="song-score">${Math.min(100, Math.round(
                        Number.isFinite(m.confidence_pct) ? m.confidence_pct : 0
                    ))}%</span>
                        ${m.spotify_url ? `<a href="${m.spotify_url}" target="_blank" style="color:#1ed760; margin-right: 5px;"><i class="fa-brands fa-spotify"></i></a>` : ''}
                        <button class="view-alignment-btn" title="View Alignment" style="background:none; border:none; color:var(--accent); padding:5px; cursor:pointer;"><i class="fa-solid fa-chart-line"></i></button>
                    </div>
                </div>
                <div class="match-row-actions">
                    <button class="btn-fb btn-fb-correct">✓ Correct</button>
                    <button class="btn-fb btn-fb-close">~ Close</button>
                </div>`;

            // Click header to inspect waveform
            li.querySelector('.match-row-header').addEventListener('click', () => {
                _updateIdentifiedSongUI(m);
                drawWaveform(m.waveform?.query_contour || [], m.waveform?.song_contour || [], m.waveform?.status || 'OK');
                // Highlight active result
                document.querySelectorAll('.match-item').forEach(item => item.classList.remove('active-inspect'));
                li.classList.add('active-inspect');
            });

            li.querySelector('.view-alignment-btn').addEventListener('click', e => {
                e.stopPropagation();
                _updateIdentifiedSongUI(m);
                drawWaveform(m.waveform?.query_contour || [], m.waveform?.song_contour || [], m.waveform?.status || 'OK');
                li.classList.add('active-inspect');
            });

            li.querySelector('.btn-fb-correct').addEventListener('click', () => {
                qbhSubmitFeedback(internalName, 'selected_from_list', {
                    melody_score: m.melody_score || 0,
                    selected_rank: i + 1,
                    top1_score: (data.top_matches[0] || {}).final_score || 0
                });

                if (typeof confetti === 'function') {
                    const count = 200;
                    const defaults = { origin: { y: 0.7 }, zIndex: 10000 };
                    function fire(ratio, opts) {
                        confetti(Object.assign({}, defaults, opts, { particleCount: Math.floor(count * ratio) }));
                    }
                    fire(0.25, { spread: 26, startVelocity: 55, });
                    fire(0.2, { spread: 60, });
                    fire(0.35, { spread: 100, decay: 0.91, scalar: 0.8 });
                    fire(0.1, { spread: 120, startVelocity: 25, decay: 0.92, scalar: 1.2 });
                    fire(0.1, { spread: 120, startVelocity: 45, });
                }

                _updateIdentifiedSongUI(m);
                document.getElementById('recorderSection')?.scrollIntoView({ behavior: 'smooth' });

                document.getElementById('waveformBlock')?.classList.add('hidden');
                qbhShowToast(`Success! ${songNameStr} is now identified.`, true);
            });

            li.querySelector('.btn-fb-close').addEventListener('click', () => {
                qbhSubmitFeedback(internalName, 'close_but_wrong', {
                    melody_score: m.melody_score || 0,
                    selected_rank: i + 1,
                    top1_score: (data.top_matches[0] || {}).final_score || 0
                });
            });
            topMatchesList.appendChild(li);
        });

        // ── Retry strip ──
        const retryExcludeBtn = document.getElementById('retryExcludeBtn');
        const retryDepthInfo = document.getElementById('retryDepthInfo');
        if (_qbhSession.retryDepth >= MAX_RETRY_DEPTH) {
            if (retryExcludeBtn) retryExcludeBtn.disabled = true;
            if (retryDepthInfo) retryDepthInfo.textContent = `Search library exhausted.`;
        } else {
            if (retryExcludeBtn) retryExcludeBtn.disabled = false;
            if (retryDepthInfo) retryDepthInfo.textContent = _qbhSession.retryDepth > 0 ? `Retry ${_qbhSession.retryDepth}/${MAX_RETRY_DEPTH} active.` : '';
        }

        // Recommendations
        datasetRecList.innerHTML = '';
        (data.similar_songs_dataset || []).forEach(s => {
            const li = document.createElement('li');
            li.className = 'dataset-match-li';
            li.innerHTML = `
                <div style="display:flex;align-items:center;gap:10px;flex:1;">
                    ${s.image ? `<img src="${s.image}" alt="" style="width:30px;border-radius:4px;">` : `<i class="fa-solid fa-compact-disc" style="width:30px;opacity:0.3;"></i>`}
                    <span class="song-name" style="font-size:0.9rem;">${s.title}</span>
                </div>
                <div class="match-stats">
                    <span class="song-score" style="font-size:0.8rem;">${s.similarity}%</span>
                </div>`;
            datasetRecList.appendChild(li);
        });

        spotifyRecList.innerHTML = '';
        (data.similar_songs_spotify || []).forEach(s => {
            const card = document.createElement('div');
            card.className = 'spotify-rec-card';
            card.innerHTML = `<img src="${s.image}" alt=""><div class="rec-info"><div class="rec-title">${s.title}</div><div class="rec-artist">${s.artist}</div></div>`;
            spotifyRecList.appendChild(card);
        });
        // Waveform comparison (Phase 11)
        if (idSong.waveform && idSong.waveform.status === 'OK' && idSong.waveform.query_contour && idSong.waveform.query_contour.length > 0) {
            drawWaveform(idSong.waveform.query_contour, idSong.waveform.song_contour, 'OK');
        } else {
            const status = idSong.waveform?.status || 'No alignment data';
            drawWaveform([], [], status);
        }
    }

    function drawWaveform(qc, sc, status = 'OK') {
        console.log(`[drawWaveform] status=${status}, qc_len=${qc?.length}, sc_len=${sc?.length}`);
        const waveformBlock = document.getElementById('waveformBlock');
        const canvas = document.getElementById('waveformCanvas');
        if (!waveformBlock || !canvas) return;

        waveformBlock.classList.remove('hidden');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (status !== 'OK') {
            ctx.fillStyle = '#ff4757';
            ctx.font = 'italic 16px Outfit';
            ctx.textAlign = 'center';

            let displayMsg = "Alignment Preview Suppressed";
            if (status === 'SUPPRESSED') {
                displayMsg = "Visualization suppressed for low-quality alignment";
            } else if (status) {
                displayMsg = status;
            }

            ctx.fillText(displayMsg, canvas.width / 2, canvas.height / 2);
            return;
        }

        const combined = qc.concat(sc || []);
        if (combined.length === 0) {
            ctx.fillStyle = '#ff4757';
            ctx.font = 'italic 12px Outfit';
            ctx.textAlign = 'center';
            ctx.fillText('No alignment path found', canvas.width / 2, canvas.height / 2);
            return;
        }
        const gMin = Math.min(...combined), gMax = Math.max(...combined);
        const gRange = (gMax - gMin) || 1;

        function drawPts(pts, col, lw) {
            if (!pts || !pts.length) return;
            ctx.beginPath(); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            pts.forEach((v, i) => {
                const x = i * (canvas.width / (pts.length - 1));
                // Enhanced scaling for better visibility of higher regions
                const y = canvas.height - 20 - ((v - gMin) / gRange) * (canvas.height - 40);
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            });
            ctx.stroke();

            // Subtle glow for the hum query (Red)
            if (col.includes('255,71,87')) {
                ctx.shadowBlur = 8; ctx.shadowColor = 'rgba(255,71,87,0.5)';
                ctx.stroke();
                ctx.shadowBlur = 0;
            }
        }
        drawPts(sc, 'rgba(30,215,96,0.3)', 2); // Song in green (thinner, background)
        drawPts(qc, 'rgba(255,71,87,1.0)', 5); // Hum in red (thicker, foreground)
    }

    // ── Retry & Reset ─────────────────────────────────────────────────
    _doRetryExcluding = async function () {
        if (_qbhSession.retryDepth >= MAX_RETRY_DEPTH) {
            qbhShowToast('Max retries reached.', false);
            return;
        }

        const accumulate = window.QBH_CONFIG?.RETRY_ACCUMULATE_EXCLUSIONS ?? false;
        if (accumulate) {
            _qbhSession.excludedSongs = [...new Set([..._qbhSession.excludedSongs, ..._qbhSession.internalNames])];
        } else {
            _qbhSession.excludedSongs = [..._qbhSession.internalNames];
        }

        _qbhSession.retryDepth++;
        _qbhSession.isRetryMode = true;

        await qbhSubmitFeedback('', 'retry_excluding_previous', 0);

        resultsSection.classList.add('hidden');
        recorderSection.classList.remove('hidden');
        resetUI();
        qbhShowToast(`Searching remaining songs...`, true);
    };

    _doFullReset = function () {
        _qbhSession = { queryId: null, internalNames: [], qType: '', excludedSongs: [], retryDepth: 0, isRetryMode: false };
        resultsSection.classList.add('hidden');
        recorderSection.classList.remove('hidden');
        resetUI();
    };

    // ── Song Details Modal ────────────────────────────────────────────
    const detailsModal = document.getElementById('detailsModal');
    const modalCloseBtn = document.getElementById('modalCloseBtn');
    const modalLoading = document.getElementById('modalLoading');
    const modalContent = document.getElementById('modalContent');

    async function fetchSongDetails(sn) {
        detailsModal.classList.remove('hidden');
        modalLoading.classList.remove('hidden');
        modalContent.classList.add('hidden');
        try {
            const res = await fetch('/song-details', { method: 'POST', headers: QBH_HEADERS, body: JSON.stringify({ song_name: sn }) });
            const data = await res.json();
            if (res.ok) showDetailsModal(data);
            else { closeDetailsModal(); qbhShowToast('Song details not found.', false); }
        } catch (e) { closeDetailsModal(); qbhShowToast('Server error.', false); }
    }

    function showDetailsModal(d) {
        modalLoading.classList.add('hidden');
        modalContent.classList.remove('hidden');
        document.getElementById('modalAlbumArt').src = d.image || '';
        document.getElementById('modalTitle').textContent = d.title || '—';
        document.getElementById('modalArtist').textContent = d.artist || '—';
        document.getElementById('modalAlbum').textContent = d.album || '—';
        document.getElementById('modalReleaseDate').textContent = d.release_date || '—';
        document.getElementById('modalDuration').textContent = d.duration || '—';
        document.getElementById('modalPopularity').textContent = `${d.popularity}/100`;

        const spLink = document.getElementById('modalSpotifyLink');
        const ytLink = document.getElementById('modalYoutubeLink');
        spLink.href = d.spotify_url || '#';
        spLink.classList.toggle('hidden', !d.spotify_url);
        ytLink.href = d.youtube_url || '#';
        ytLink.classList.toggle('hidden', !d.youtube_url);

        const simList = document.getElementById('modalSimilarList');
        simList.innerHTML = '';
        (d.similar_tracks || []).forEach(s => {
            const card = document.createElement('div');
            card.className = 'modal-sim-card';
            card.innerHTML = `
                <img src="${s.image}" alt="">
                <div class="sim-info"><div class="sim-title">${s.title}</div><div class="sim-artist">${s.artist}</div></div>`;
            simList.appendChild(card);
        });
    }

    function closeDetailsModal() { detailsModal.classList.add('hidden'); }
    modalCloseBtn.addEventListener('click', closeDetailsModal);
    detailsModal.addEventListener('click', e => { if (e.target === detailsModal) closeDetailsModal(); });

    function showError(msg) {
        loadingSection.classList.add('hidden');
        recorderSection.classList.remove('hidden');
        resetUI();
        statusText.textContent = `❌ ${msg}`;
        statusText.style.color = '#ff4757';
    }

    recordBtn.addEventListener('click', () => { if (!isRecording) startRecording(); else stopRecording(); });
});
