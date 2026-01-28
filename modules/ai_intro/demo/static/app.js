/**
 * AI Concepts Demo - Frontend Application
 */

// Determine API base URL
// If served from the backend port (8001), use relative paths.
// If served from file://, default to localhost:8001.
// Otherwise (e.g., served from 8000), point to same hostname but port 8001.
const BACKEND_PORT = '8001';
let apiBase = '';

if (window.location.protocol === 'file:') {
    apiBase = `http://localhost:${BACKEND_PORT}`;
} else if (window.location.port !== BACKEND_PORT) {
    apiBase = `${window.location.protocol}//${window.location.hostname}:${BACKEND_PORT}`;
}

const API_BASE = apiBase;

// =============================================================================
// Tab Navigation
// =============================================================================

function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    const panels = document.querySelectorAll('.panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.dataset.tab;

            // Update tabs
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Update panels
            panels.forEach(p => p.classList.remove('active'));
            document.getElementById(`panel-${tabId}`).classList.add('active');
        });
    });
}

function initSubTabs() {
    const subTabs = document.querySelectorAll('.sub-tab');

    subTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const subtabId = tab.dataset.subtab;
            const parent = tab.closest('.panel');

            // Update sub-tabs
            parent.querySelectorAll('.sub-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Update sub-panels
            parent.querySelectorAll('.sub-panel').forEach(p => p.classList.remove('active'));
            document.getElementById(`subpanel-${subtabId}`).classList.add('active');
        });
    });
}

// =============================================================================
// API Helpers
// =============================================================================

async function checkHealth() {
    const status = document.getElementById('status');
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();

        if (data.status === 'ok') {
            status.classList.add('connected');
            status.classList.remove('error');
            status.querySelector('.status-text').textContent = 'Connected';
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        status.classList.add('error');
        status.classList.remove('connected');
        status.querySelector('.status-text').textContent = 'Disconnected';
        console.error('Health check failed:', error);
    }
}

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

// =============================================================================
// LLM Demo
// =============================================================================

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
                previewEl.innerHTML = marked.parse(rawContent);
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
                        previewEl.innerHTML = marked.parse(rawContent);
                    }
                }
            });
            rawContent = outputEl.textContent;
            if (isPreviewMode) {
                previewEl.innerHTML = marked.parse(rawContent);
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

// =============================================================================
// Embeddings Demo
// =============================================================================

function initEmbeddings() {
    // Compare
    const text1Input = document.getElementById('embed-text1');
    const text2Input = document.getElementById('embed-text2');
    const compareBtn = document.getElementById('embed-compare');
    const compareOutput = document.getElementById('embed-compare-output');

    compareBtn.addEventListener('click', async () => {
        const text1 = text1Input.value.trim();
        const text2 = text2Input.value.trim();
        if (!text1 || !text2) return;

        compareBtn.disabled = true;
        compareOutput.innerHTML = '<span class="thought">Computing embeddings...</span>';

        try {
            const url = `${API_BASE}/api/embeddings/compare?text1=${encodeURIComponent(text1)}&text2=${encodeURIComponent(text2)}`;
            const response = await fetch(url);
            const data = await response.json();

            const percentage = Math.round(data.similarity * 100);
            compareOutput.innerHTML = `
                <div class="similarity-result">
                    <div class="similarity-score">${percentage}%</div>
                    <span class="similarity-label">${data.interpretation}</span>
                    <div class="similarity-bar">
                        <div class="similarity-fill" style="width: ${percentage}%"></div>
                    </div>
                </div>
            `;
        } catch (error) {
            compareOutput.innerHTML = `<span class="error">Error: ${error.message}</span>`;
        } finally {
            compareBtn.disabled = false;
        }
    });

    // Search
    const queryInput = document.getElementById('embed-query');
    const searchBtn = document.getElementById('embed-search');
    const searchOutput = document.getElementById('embed-search-output');

    searchBtn.addEventListener('click', async () => {
        const query = queryInput.value.trim();
        if (!query) return;

        searchBtn.disabled = true;
        searchOutput.innerHTML = '<span class="thought">Searching...</span>';

        try {
            const url = `${API_BASE}/api/embeddings/search?query=${encodeURIComponent(query)}`;
            const response = await fetch(url);
            const data = await response.json();

            let html = '<ul class="search-results">';
            for (const result of data.results) {
                const percentage = Math.round(result.similarity * 100);
                html += `
                    <li class="search-result">
                        <span class="score">${percentage}%</span>
                        <p class="text">${result.document}</p>
                    </li>
                `;
            }
            html += '</ul>';
            searchOutput.innerHTML = html;
        } catch (error) {
            searchOutput.innerHTML = `<span class="error">Error: ${error.message}</span>`;
        } finally {
            searchBtn.disabled = false;
        }
    });
}

// =============================================================================
// RAG Demo
// =============================================================================

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

// =============================================================================
// Agent Demo
// =============================================================================

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

// =============================================================================
// Settings
// =============================================================================

let availableModels = [];

async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/api/models`);
        const data = await response.json();
        availableModels = data.models || [];
        return {
            models: availableModels,
            currentLlm: data.current_llm,
            currentEmbed: data.current_embed
        };
    } catch (error) {
        console.error('Failed to load models:', error);
        return { models: [], currentLlm: '', currentEmbed: '' };
    }
}

function initSettings() {
    const settingsBtn = document.getElementById('settings-btn');
    const settingsModal = document.getElementById('settings-modal');
    const settingsClose = document.getElementById('settings-close');
    const settingsSave = document.getElementById('settings-save');
    const settingsReset = document.getElementById('settings-reset');
    const llmSelect = document.getElementById('llm-model-select');
    const embedSelect = document.getElementById('embed-model-select');
    const currentConfigEl = document.getElementById('current-config');

    async function openSettings() {
        settingsModal.classList.remove('hidden');

        const { models, currentLlm, currentEmbed } = await loadModels();

        // Populate LLM models
        const llmModels = models.filter(m => m.type === 'llm');
        llmSelect.innerHTML = llmModels.map(m =>
            `<option value="${m.name}" ${m.name === currentLlm ? 'selected' : ''}>${m.name} (${m.size_gb}GB)</option>`
        ).join('');

        // Populate embedding models
        const embedModels = models.filter(m => m.type === 'embedding');
        embedSelect.innerHTML = embedModels.map(m =>
            `<option value="${m.name}" ${m.name === currentEmbed ? 'selected' : ''}>${m.name} (${m.size_gb}GB)</option>`
        ).join('');

        // If no embedding models found, show all models
        if (embedModels.length === 0) {
            embedSelect.innerHTML = models.map(m =>
                `<option value="${m.name}" ${m.name === currentEmbed ? 'selected' : ''}>${m.name}</option>`
            ).join('');
        }

        currentConfigEl.innerHTML = `Current: <strong>${currentLlm}</strong> (LLM) | <strong>${currentEmbed}</strong> (Embed)`;
    }

    function closeSettings() {
        settingsModal.classList.add('hidden');
    }

    async function saveSettings() {
        const llmModel = llmSelect.value;
        const embedModel = embedSelect.value;

        try {
            const response = await fetch(
                `${API_BASE}/api/config?llm_model=${encodeURIComponent(llmModel)}&embed_model=${encodeURIComponent(embedModel)}`,
                { method: 'POST' }
            );
            const data = await response.json();

            // Update footer with new models
            updateFooterModels(data.llm_model, data.embed_model);

            closeSettings();
            checkHealth(); // Refresh status
        } catch (error) {
            console.error('Failed to save settings:', error);
            alert('Failed to save settings');
        }
    }

    async function resetSettings() {
        try {
            const configResponse = await fetch(`${API_BASE}/api/config`);
            const config = await configResponse.json();

            const response = await fetch(
                `${API_BASE}/api/config?llm_model=${encodeURIComponent(config.defaults.llm_model)}&embed_model=${encodeURIComponent(config.defaults.embed_model)}`,
                { method: 'POST' }
            );
            const data = await response.json();

            updateFooterModels(data.llm_model, data.embed_model);
            closeSettings();
            checkHealth();
        } catch (error) {
            console.error('Failed to reset settings:', error);
        }
    }

    function updateFooterModels(llm, embed) {
        const footer = document.querySelector('.footer');
        const modelSpan = footer.querySelector('span:last-child');
        if (modelSpan) {
            modelSpan.textContent = `${llm.split(':')[0]} + ${embed.split(':')[0]}`;
        }
    }

    settingsBtn.addEventListener('click', openSettings);
    settingsClose.addEventListener('click', closeSettings);
    settingsSave.addEventListener('click', saveSettings);
    settingsReset.addEventListener('click', resetSettings);

    // Close on overlay click
    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) closeSettings();
    });

    // Close on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !settingsModal.classList.contains('hidden')) {
            closeSettings();
        }
    });
}

// =============================================================================
// Initialize
// =============================================================================

function initBanner() {
    const closeBtn = document.getElementById('banner-close-btn');
    const banner = document.getElementById('setup-banner');

    if (closeBtn && banner) {
        closeBtn.addEventListener('click', () => {
            banner.style.display = 'none';
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initBanner();
    initTabs();
    initSubTabs();
    initLLM();
    initEmbeddings();
    initRAG();
    initAgent();
    initSettings();

    // Check health on load and periodically
    checkHealth();
    setInterval(checkHealth, 30000);
});
