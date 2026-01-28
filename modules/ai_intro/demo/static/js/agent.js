/**
 * Agent Demo Panel
 */

import { API_BASE } from './config.js';
import { streamSSE } from './api.js';

function initAgent() {
    const queryInput = document.getElementById('agent-query');
    const runBtn = document.getElementById('agent-run');
    const outputEl = document.getElementById('agent-output');
    const clearBtn = document.getElementById('agent-clear');

    runBtn.addEventListener('click', async () => {
        const query = queryInput.value.trim();
        if (!query) return;

        outputEl.innerHTML = '';
        runBtn.disabled = true;

        try {
            const url = `${API_BASE}/api/agent/run?query=${encodeURIComponent(query)}`;
            await streamSSE(url, outputEl);
        } catch (error) {
            console.error('Agent run failed:', error);
        } finally {
            runBtn.disabled = false;
        }
    });

    clearBtn.addEventListener('click', () => {
        outputEl.innerHTML = '';
    });
}

export { initAgent };
