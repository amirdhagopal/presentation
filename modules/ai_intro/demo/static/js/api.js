/**
 * API Helper Functions
 */

import { API_BASE } from './config.js';

/**
 * Check API health status
 */
async function checkHealth() {
    const status = document.getElementById('status');
    const banner = document.getElementById('setup-banner');

    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();

        if (data.status === 'ok') {
            status.classList.add('connected');
            status.classList.remove('error');
            status.querySelector('.status-text').textContent = 'Connected';
            if (banner) banner.classList.add('hidden');
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        status.classList.add('error');
        status.classList.remove('connected');
        status.querySelector('.status-text').textContent = 'Disconnected';
        console.error('Health check failed:', error);
        if (banner) banner.classList.remove('hidden');
    }
}

/**
 * Stream Server-Sent Events from a URL
 */
function streamSSE(url, outputEl, options = {}) {
    return new Promise((resolve, reject) => {
        const eventSource = new EventSource(url);
        let fullText = '';

        if (options.onStart) options.onStart();

        eventSource.onmessage = (event) => {
            const data = event.data;

            if (data === '[DONE]') {
                eventSource.close();
                if (options.onComplete) options.onComplete(fullText);
                resolve(fullText);
                return;
            }

            if (data.startsWith('[ERROR]')) {
                eventSource.close();
                const error = data.replace('[ERROR] ', '');
                outputEl.innerHTML += `<span class="error">${error}</span>\n`;
                reject(new Error(error));
                return;
            }

            if (data.startsWith('[STEP]')) {
                const step = data.replace('[STEP] ', '');
                outputEl.innerHTML += `<span class="step">📍 ${step}</span>\n`;
            } else if (data.startsWith('[CONTEXT]')) {
                const ctx = data.replace('[CONTEXT] ', '');
                outputEl.innerHTML += `<span class="context">${ctx}</span>\n`;
            } else if (data.startsWith('[THOUGHT]')) {
                const thought = data.replace('[THOUGHT] ', '');
                outputEl.innerHTML += `<span class="thought">💭 ${thought}</span>\n`;
            } else if (data.startsWith('[ACTION]')) {
                const action = data.replace('[ACTION] ', '');
                outputEl.innerHTML += `<span class="action">🔧 ${action}</span>\n`;
            } else if (data.startsWith('[OBSERVATION]')) {
                const obs = data.replace('[OBSERVATION] ', '');
                outputEl.innerHTML += `<span class="observation">👁️ ${obs}</span>\n`;
            } else if (data.startsWith('[ANSWER]')) {
                const answer = data.replace('[ANSWER] ', '');
                outputEl.innerHTML += `<span class="answer">✅ ${answer}</span>\n`;
            } else {
                fullText += data;
                outputEl.textContent = fullText;
            }

            // Auto-scroll
            outputEl.scrollTop = outputEl.scrollHeight;
        };

        eventSource.onerror = (error) => {
            eventSource.close();
            reject(error);
        };
    });
}

export { checkHealth, streamSSE };
