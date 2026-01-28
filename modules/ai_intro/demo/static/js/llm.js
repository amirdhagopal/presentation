/**
 * LLM Demo Panel
 */

import { API_BASE } from './config.js';
import { streamSSE } from './api.js';

function initLLM() {
    const promptInput = document.getElementById('llm-prompt');
    const tempSlider = document.getElementById('llm-temp');
    const tempValue = document.getElementById('llm-temp-value');
    const generateBtn = document.getElementById('llm-generate');
    const outputEl = document.getElementById('llm-output');
    const previewEl = document.getElementById('llm-preview');
    const clearBtn = document.getElementById('llm-clear');
    const previewToggle = document.getElementById('llm-preview-toggle');

    let rawContent = '';
    let isPreviewMode = false;

    tempSlider.addEventListener('input', () => {
        tempValue.textContent = tempSlider.value;
    });

    // Toggle between raw and preview mode
    previewToggle.addEventListener('click', () => {
        isPreviewMode = !isPreviewMode;
        previewToggle.classList.toggle('active', isPreviewMode);

        if (isPreviewMode) {
            // Show markdown preview
            outputEl.classList.add('hidden');
            previewEl.classList.remove('hidden');
            if (rawContent) {
                previewEl.innerHTML = marked.parse(rawContent, { breaks: true });
            }
        } else {
            // Show raw output
            previewEl.classList.add('hidden');
            outputEl.classList.remove('hidden');
        }
    });

    generateBtn.addEventListener('click', async () => {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        const temp = parseFloat(tempSlider.value);
        rawContent = '';
        outputEl.textContent = '';
        previewEl.innerHTML = '';
        generateBtn.disabled = true;

        try {
            const url = `${API_BASE}/api/llm/generate?prompt=${encodeURIComponent(prompt)}&temperature=${temp}`;
            await streamSSE(url, outputEl, {
                onComplete: (text) => {
                    rawContent = text;
                    if (isPreviewMode) {
                        previewEl.innerHTML = marked.parse(rawContent, { breaks: true });
                    }
                }
            });
            rawContent = outputEl.textContent;
            if (isPreviewMode) {
                previewEl.innerHTML = marked.parse(rawContent, { breaks: true });
            }
        } catch (error) {
            console.error('Generation failed:', error);
        } finally {
            generateBtn.disabled = false;
        }
    });

    clearBtn.addEventListener('click', () => {
        rawContent = '';
        outputEl.textContent = '';
        previewEl.innerHTML = '';
    });
}

export { initLLM };
