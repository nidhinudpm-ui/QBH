document.addEventListener('DOMContentLoaded', () => {
    const recordBtn = document.getElementById('recordBtn');
    const recorderSection = document.getElementById('recorderSection');
    const visualizerContainer = document.getElementById('visualizerContainer');
    const statusText = document.getElementById('statusText');
    const timerText = document.getElementById('timerText');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const resultsSection = document.getElementById('resultsSection');
    const loadingSection = document.getElementById('loadingSection');
    const retryBtn = document.getElementById('retryBtn');

    // Result elements
    const albumArt = document.getElementById('albumArt');
    const songTitle = document.getElementById('songTitle');
    const songArtist = document.getElementById('songArtist');
    const songAlbum = document.getElementById('songAlbum');
    const songDate = document.getElementById('songDate');
    const songConfidence = document.getElementById('songConfidence');
    const previewPlayer = document.getElementById('previewPlayer');
    const spotifyLink = document.getElementById('spotifyLink');
    const topMatchesList = document.getElementById('topMatchesList');
    const datasetRecList = document.getElementById('datasetRecList');
    const datasetRecBlock = document.getElementById('datasetRecBlock');
    const spotifyRecList = document.getElementById('spotifyRecList');
    const spotifyRecBlock = document.getElementById('spotifyRecBlock');

    // ─── Audio Recording (Web Audio API → WAV) ──────────────────────
    let audioContext, sourceNode, scriptProcessor, stream;
    let pcmSamples = [];
    let isRecording = false;
    let timerInterval, timeElapsed = 0;
    const TARGET_SR = 22050, MAX_SECONDS = 10, BUFFER_SIZE = 4096;

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

    async function startRecording() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: TARGET_SR
                }
            });
        }
        catch { statusText.textContent = "Microphone access denied."; statusText.style.color = "#ff3333"; return; }

        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SR });
        sourceNode = audioContext.createMediaStreamSource(stream);
        scriptProcessor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);
        pcmSamples = [];

        scriptProcessor.onaudioprocess = e => {
            if (!isRecording) return;
            const raw = e.inputBuffer.getChannelData(0);
            const rs = downsample(raw, audioContext.sampleRate, TARGET_SR);
            for (const s of rs) pcmSamples.push(s);
        };

        sourceNode.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);
        isRecording = true;
        visualizerContainer.classList.add('recording');
        statusText.textContent = "Recording… tap to stop";
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

    // ─── API Call ────────────────────────────────────────────────────
    async function sendAudio(blob) {
        const fd = new FormData();
        fd.append('audio', blob, 'recording.wav');
        try {
            const res = await fetch('/identify-song', { method: 'POST', body: fd });
            const data = await res.json();
            if (res.ok && data.success) displayResults(data);
            else showError(data.error || 'Matching failed.');
        } catch (e) { console.error(e); showError('Server connection failed.'); }
    }

    // ─── Render Results ─────────────────────────────────────────────
    function displayResults(data) {
        loadingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        const song = data.identified_song || {};

        // Album art
        if (song.image) { albumArt.src = song.image; albumArt.classList.remove('hidden'); }
        else albumArt.classList.add('hidden');

        // Metadata
        songTitle.textContent = song.title || '—';
        songArtist.textContent = song.artist || 'Unknown artist';
        songAlbum.textContent = song.album || '';
        songDate.textContent = song.release_date || '';
        songConfidence.textContent = `${song.confidence || 0}% Confidence`;

        // Preview player
        if (song.preview_url) {
            previewPlayer.src = song.preview_url;
            previewPlayer.classList.remove('hidden');
        } else previewPlayer.classList.add('hidden');

        // Spotify link
        if (song.spotify_url) {
            spotifyLink.href = song.spotify_url;
            spotifyLink.classList.remove('hidden');
        } else spotifyLink.classList.add('hidden');

        // Top matches
        topMatchesList.innerHTML = '';
        (data.top_matches || []).forEach((m, i) => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span class="song-name">
                    <i class="fa-solid fa-${i === 0 ? 'trophy' : 'music'}" style="margin-right:8px;opacity:.5;"></i>
                    ${m.title || m.song_name || '—'}
                    ${m.artist ? `<span style="color:var(--text-muted);font-size:.8rem;"> — ${m.artist}</span>` : ''}
                </span>
                <span class="song-score">${m.confidence}%</span>`;
            topMatchesList.appendChild(li);
        });

        // Dataset recommendations
        datasetRecList.innerHTML = '';
        const dsRecs = data.similar_songs_dataset || [];
        if (dsRecs.length) {
            datasetRecBlock.classList.remove('hidden');
            dsRecs.forEach(s => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <span class="song-name"><i class="fa-solid fa-compact-disc" style="margin-right:8px;opacity:.4;"></i>${s.title || s.song_name || ''}</span>
                    <span class="song-score">${s.similarity}%</span>`;
                datasetRecList.appendChild(li);
            });
        } else datasetRecBlock.classList.add('hidden');

        // Spotify recommendations
        spotifyRecList.innerHTML = '';
        const spRecs = data.similar_songs_spotify || [];
        if (spRecs.length) {
            spotifyRecBlock.classList.remove('hidden');
            spRecs.forEach(s => {
                const card = document.createElement('div');
                card.className = 'spotify-rec-card';
                card.innerHTML = `
                    ${s.image ? `<img src="${s.image}" alt="">` : ''}
                    <div class="rec-info">
                        <div class="rec-title">${s.title}</div>
                        <div class="rec-artist">${s.artist}</div>
                    </div>
                    ${s.spotify_url ? `<a href="${s.spotify_url}" target="_blank"><i class="fa-brands fa-spotify"></i></a>` : ''}`;
                spotifyRecList.appendChild(card);
            });
        } else spotifyRecBlock.classList.add('hidden');
    }

    function showError(msg) {
        loadingSection.classList.add('hidden');
        recorderSection.classList.remove('hidden');
        resetUI();
        statusText.textContent = `❌ ${msg}`;
        statusText.style.color = "#ff3333";
    }

    // ─── Events ─────────────────────────────────────────────────────
    recordBtn.addEventListener('click', () => { if (!isRecording) startRecording(); else stopRecording(); });
    retryBtn.addEventListener('click', () => { resultsSection.classList.add('hidden'); recorderSection.classList.remove('hidden'); resetUI(); });
});
