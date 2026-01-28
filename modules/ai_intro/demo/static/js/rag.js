/**
 * RAG Demo Panel
 */

import { API_BASE } from './config.js';
import { streamSSE } from './api.js';

function initRAG() {
    const questionInput = document.getElementById('rag-question');
    const askBtn = document.getElementById('rag-ask');
    const outputEl = document.getElementById('rag-output');
    const clearBtn = document.getElementById('rag-clear');

    askBtn.addEventListener('click', async () => {
        const question = questionInput.value.trim();
        if (!question) return;

        outputEl.innerHTML = '';
        askBtn.disabled = true;

        try {
            const url = `${API_BASE}/api/rag/query?question=${encodeURIComponent(question)}`;
            await streamSSE(url, outputEl);
        } catch (error) {
            console.error('RAG query failed:', error);
        } finally {
            askBtn.disabled = false;
        }
    });

    clearBtn.addEventListener('click', () => {
        outputEl.innerHTML = '';
    });
}

export { initRAG };
